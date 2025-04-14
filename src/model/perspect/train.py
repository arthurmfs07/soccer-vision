import cv2
import torch
from torch.nn import functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

from src.logger import setup_logger
from src.model.detect.objdetect import ObjectDetector
from src.model.perspect.cnn import CNN
from src.struct.detection import Detection
from src.utils import get_actual_yolo, get_csv_path, get_data_path
from src.model.perspect.batch import Batch
from src.visual.field import PitchConfig

@dataclass
class TrainConfig:
    save_path: Path = None # field(default_factory=load_abs_path() / "perspect_cnn.pth")
    epochs: int = 50
    batch_size: int = 16
    lr: float = 6e-7
    device: str = "cuda"


class PerspectTrainer:
    def __init__(self):
        self.config = TrainConfig()
        self.logger = setup_logger("train.log")

        self.epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.save_path = self.config.save_path
        self.device = self.config.device

        self.data_dir = get_data_path()
        self.csv_path = get_csv_path() 
        self.yolo_path = get_actual_yolo()

        self.batches = Batch(
            self.data_dir, self.csv_path, 
            batch_size=self.batch_size
            )

        self.object_detect = ObjectDetector(
                model_path=self.yolo_path,
                device=self.device
            )

        self.cnn = CNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.config.lr)

        self.current_image_shape: Optional[Tuple[int]] = None


    def hungarian_loss(self, yhat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Hungarian-matched MSE loss.
        yhat:   [B, N, 2] - predicted positions
        labels: [B, N, 2] - ground truth positions
        """
        B, N, _ = yhat.shape
        total_loss = 0.0
        for b in range(B):
            pred = yhat[b]
            target = labels[b]

            # Filter out padding (-1 markers)
            pred = pred[(pred != -1).all(dim=1)]
            target = target[(target != -1).all(dim=1)]

            if pred.size(0) == 0 or target.size(0) == 0:
                continue

            cost = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0) # [M, N]
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

            matched_pred = pred[row_ind]
            matched_target = target[col_ind]

            total_loss += F.mse_loss(matched_pred, matched_target)

        return total_loss / B


    def normalize_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centroid = points.mean(dim=0)
        diff = points - centroid
        dists = torch.norm(diff, dim=1)
        mean_dist = dists.mean()
        scale = torch.sqrt(torch.tensor(2.0, device=points.device)) / (mean_dist + 1e-8)
        T = torch.tensor([[scale, 0, -scale * centroid[0]],
                          [0, scale, -scale * centroid[1]],
                          [0, 0, 1]], dtype=torch.float32, device=points.device)
        ones = torch.ones((points.shape[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1)  # [4,3]
        points_norm_h = (T @ points_h.t()).t()  # [4,3]
        points_norm = points_norm_h[:, :2] / points_norm_h[:, 2:3]
        return T, points_norm


    def solve_homography(self, predicted_square: torch.Tensor) -> torch.Tensor:
        """
        predicted_square: [B, 8], CNN output
        returns H : [B, 3, 3]
        """
        if self.current_image_shape is None:
            raise ValueError("Current image shape not set. Call this method after a forward pass.")
        B, C, H_img, W_img = self.current_image_shape
        s = H_img / 3.0
        center_x = W_img / 2.0
        center_y = 2 * H_img / 3.0
        # Create the base square in video space (assume same for entire batch)
        base_square = torch.tensor([
            [center_x - s / 2.0, center_y - s / 2.0],  # top-left
            [center_x + s / 2.0, center_y - s / 2.0],  # top-right
            [center_x + s / 2.0, center_y + s / 2.0],  # bottom-right
            [center_x - s / 2.0, center_y + s / 2.0]   # bottom-left
        ], dtype=torch.float32, device=self.device)  # [4, 2]
        base_square = base_square.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 2]
        
        dst = predicted_square.view(B, 4, 2)
        
        H_matrices = []
        for b in range(B):
            src = base_square[b]    # [4,2]
            dst_b = dst[b]          # [4,2]
            T_src, src_norm = self.normalize_points(src)
            T_dst, dst_norm = self.normalize_points(dst_b)
            A_rows = []
            b_rows = []
            for i in range(4):
                x, y = src_norm[i]
                x_dst, y_dst = dst_norm[i]
                row1 = torch.tensor([x, y, 1, 0, 0, 0, -x * x_dst, -y * x_dst], 
                                      dtype=torch.float32, device=self.device)
                row2 = torch.tensor([0, 0, 0, x, y, 1, -x * y_dst, -y * y_dst],
                                      dtype=torch.float32, device=self.device)
                A_rows.append(row1)
                A_rows.append(row2)
                b_rows.append(x_dst)
                b_rows.append(y_dst)
            A = torch.stack(A_rows, dim=0)  # [8,8]
            b_vec = torch.stack(b_rows, dim=0)  # [8]
            
            reg = 1e-4
            I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
            A_reg = A + reg * I.expand_as(A)
            h_norm = torch.linalg.solve(A_reg, b_vec)
            H_norm = torch.zeros((3,3), dtype=torch.float32, device=self.device)
            H_norm[0, :3] = h_norm[0:3]
            H_norm[1, :3] = h_norm[3:6]
            H_norm[2, :2] = h_norm[6:8]
            H_norm[2,2] = 1.0
            H_b = torch.linalg.inv(T_dst) @ H_norm @ T_src
            H_matrices.append(H_b)
        H_matrix = torch.stack(H_matrices, dim=0)  # [B, 3, 3]
        return H_matrix

    def apply_homography(self, H: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Apply homography transformation
        H: [B, 3, 3]
        points: [B, N, 2]
        Returns transfomed points [B, N, 2]
        """
        B, N, _ = points.shape
        ones = torch.ones((B, N, 1), device=points.device)
        points_h = torch.cat([points, ones], dim=2) # [B, N, 3]
        points_transformed = torch.bmm(points_h, H.transpose(1, 2)) # [B, N, 3]
        points_transformed = points_transformed[:, :, :2] / points_transformed[:, :, 2:]
        return points_transformed[:, :, :2] # [B, N, 2]
    

    def extract_detection_points(self, detections: List[Detection], max_points: int = 30) -> torch.Tensor:
        batch_points = []
        for det in detections:
            boxes = det.boxes # [N, 4]
            points = []
            for box in boxes:
                x1, y1, x2, y2 = box
                foot = [(x1 + x2) / 2, y2]
                points.append(foot)
            if points:
                pts_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
            else:
                pts_tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            num_points = pts_tensor.shape[0]
            if num_points < max_points:
                padding = torch.full((max_points - num_points, 2), fill_value=-1, device=self.device)
                pts_tensor = torch.cat([pts_tensor, padding], dim=0)
            elif num_points > max_points:
                pts_tensor = pts_tensor[:max_points, :]
            batch_points.append(pts_tensor)
        return torch.stack(batch_points, dim=0) # [B, max_points, 2]
    

    def train_batches(self, epochs: int = -1):
        """
        Train the CNN on the dataset using batches. for n epochs 
        (use config if epochs = -1)

        data_dir: organized as
            `match_<match-id>/match<match-id>_<timestamp>.jpg`
        csv_path:
            columns: [timestamp,freeze_frame,match_id,period]
        """
        self.logger.info("Starting training...")
        self.cnn.train()
        epochs = self.epochs if epochs == -1 else epochs

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            n = 0

            self.batches.__iter__()

            for batch_idx, batch in enumerate(self.batches):
                images = batch.image.to(self.device)
                labels = batch.label.to(self.device)

                self.current_image_shape = images.shape

                detections = self.object_detect.detect(images)
                detections_pts = self.extract_detection_points(detections, max_points=labels.shape[1])
                
                predicted_square = self.cnn(images)
                H = self.solve_homography(predicted_square)

                yhat_norm = self.apply_homography(H, detections_pts)
                labels_norm = self._field_norm(labels)

                yhat_field_model = self._field_denorm(yhat_norm)

                # loss_main = self.hungarian_loss(yhat_norm, labels_norm)
                loss = F.mse_loss(yhat_norm, labels_norm)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n += 1
                self.logger.info(f"[EPOCH {epoch}] - Batch {batch_idx} Loss: {loss.item():.4f}")

                yield (epoch, batch_idx, images, labels, detections, yhat_field_model, loss.item())

            avg_loss = epoch_loss / max(n, 1)
            self.logger.info(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

            if self.save_path:
                self.save_model()
                self.logger.info(f"Model saved to {self.save_path}")


    def train(self, epochs: int = 1):
        """
        Standard training method that uses the train_batches() generator
        but does not yield.
        """
        for _ in self.train_batches(epochs):
            pass
        self.logger.info("Training completed.")


    def inference(self, max_batches: int = 10) -> List[tuple]:
        """
        Runs inference to get predicted homographies and transformed detections.
        Returns a list of (yhat, label) tuples.
        """
        self.logger.info("Running inference...")
        self.cnn.eval()
        results = []

        with torch.no_grad():
            for i, batch in enumerate(self.batches):
                if i >= max_batches:
                    break

                images = batch.image.to(self.device)
                labels = batch.label.to(self.device)

                detections = self.object_detect.detect(images)
                detections_pts = self.extract_detection_points(detections, max_points=labels.shape[1])

                predicted_square = self.cnn(images)
                H = self.solve_homography(predicted_square)

                yhat = self.apply_homography(H, detections_pts)

                results.append((yhat.cpu(), labels.cpu()))

        return results


    def evaluate(self, max_batches: int = 10) -> None:
        """
        Evaluate model performance using average MSE on inferred positions.
        """
        results = self.inference(max_batches=max_batches)

        total_loss = 0.0
        count = 0

        for i, (yhat, label) in enumerate(results):
            mask = (label != -1).all(dim=2) # [B, N]
            if not mask.any():
                continue
            masked_yhat = yhat[mask]
            masked_label = label[mask]
            mse = F.mse_loss(masked_yhat, masked_label).item()
            total_loss += mse
            count += 1

            if i == 0:
                self.logger.info(f"First batch MSE: {mse:.4f}")

        avg_loss = total_loss / max(count, 1)
        self.logger.info(f"Average Evaluation Loss: {avg_loss:.4f}")


    def _field_norm(self, points):
        points_norm = points.clone()
        points_norm[..., 0] /= PitchConfig().length
        points_norm[..., 1] /= PitchConfig().width
        return points_norm

    def _field_denorm(self, points_norm):
        points = points_norm.clone()
        points[..., 0] *= PitchConfig().length
        points[..., 1] *= PitchConfig().width
        return points


    def save_model(self) -> None:
        """Saves the trained model."""
        torch.save(self.cnn.state_dict(), self.save_path)
        self.logger.info(f"Model saved to {self.save_path}")





if __name__ == "__main__":
        
    trainer = PerspectTrainer()

    trainer.train(10)
    trainer.evaluate()
    trainer.save_model()
