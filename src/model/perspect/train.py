import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Optional, Tuple, Generator
from scipy.optimize import linear_sum_assignment

from torch.utils.data import Dataset, DataLoader

from src.logger import setup_logger
from src.model.detect.objdetect import ObjectDetector
from src.model.perspect.cnn import CNN
from src.struct.detection import Detection
from src.utils import get_actual_yolo, get_csv_path, get_data_path
from src.model.perspect.batch import Batch
from src.visual.field import PitchConfig


@dataclass
class TrainConfig:
    dataset_folder: Path
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    device: str = "cuda"
    save_path: Optional[Path] = None
    train_on_homography: bool = True
    train_on_player_position: bool = False



class HomographyDataset(Dataset):
    """
    Dataset for homography training.
    Expects files named as "annot_frame_<index>.png" and
    "annot_frame_<index>_H.npy" in the dataset_folder.
    Since the CNN is now adaptive, images are returned in full resolution.
    """
    def __init__(self, dataset_folder: Path) -> None:
        self.dataset_folder = dataset_folder
        self.image_paths = sorted(glob.glob(str(dataset_folder / "annot_frame_*.png")))
        self.pairs = []
        for img_path in self.image_paths:
            base = Path(img_path).stem  # e.g., "annot_frame_0"
            # Skip files that are homography matrices (ending with _H)
            if base.endswith("_H"):
                continue
            parts = base.split("_")
            if len(parts) < 3:
                continue
            idx = parts[2]
            h_path = dataset_folder / f"annot_frame_{idx}_H.npy"
            if h_path.exists():
                self.pairs.append((Path(img_path), h_path))
        if not self.pairs:
            raise ValueError(f"No valid homography pairs found in {dataset_folder}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.Tensor]:
        img_path, H_path = self.pairs[index]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # Convert BGR to RGB and normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Convert to a torch tensor of shape [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        H_gt = np.load(str(H_path))
        H_gt_tensor = torch.tensor(H_gt, dtype=torch.float32)
        return img_tensor, H_gt_tensor



class PerspectTrainer:
    """
    Trainer for two-stage training of a perspective homography CNN.
    The first stage (if enabled) overfits on annotated homography pairs.
    The second stage (if enabled) takes YOLO detections, applies the estimated homography,
    and compares the projected player positions with ground truth labels.
    
    Modular functions are provided to compute the base square, solve for H using DLT,
    apply homography, and extract detection points.
    Normalization of points or homography is removed.
    """
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.logger = setup_logger("train.log")
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.save_path = config.save_path
        self.device = config.device

        self.data_dir = get_data_path()
        self.csv_path = get_csv_path()
        self.yolo_path = get_actual_yolo()

        # Create the player positions dataset loader using the existing Batch class.
        self.batches = Batch(
            self.data_dir, self.csv_path, 
            batch_size=self.batch_size
        )
        # Create YOLO detector for player position training.
        self.object_detect = ObjectDetector(
            model_path=self.yolo_path,
            device=self.device
        )
        # Create homography training dataset and loader.
        self.homography_dataset = HomographyDataset(self.config.dataset_folder)
        self.homography_loader = DataLoader(
            self.homography_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        # Initialize CNN and optimizer.
        self.cnn = CNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.config.lr)
        self.current_image_shape: Optional[Tuple[int]] = None

    def hungarian_loss(self, yhat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Hungarian-matched MSE loss.
        yhat: [B, N, 2] predicted positions.
        labels: [B, N, 2] ground truth positions.
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
            cost = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            matched_pred = pred[row_ind]
            matched_target = target[col_ind]
            total_loss += F.mse_loss(matched_pred, matched_target)
        return total_loss / B

    def _create_base_square(self, image_shape: Tuple[int, int, int, int], as_tensor: bool = True) -> torch.Tensor:
        """
        Compute the base square from image dimensions.
        image_shape: [B, C, H, W]
        Returns a tensor of shape [B, 4, 2].
        """
        B, C, H_img, W_img = image_shape
        s = H_img / 3.0
        center_x = W_img / 2.0
        center_y = 2 * H_img / 3.0
        base_square = [
            [center_x - s / 2.0, center_y - s / 2.0],
            [center_x + s / 2.0, center_y - s / 2.0],
            [center_x + s / 2.0, center_y + s / 2.0],
            [center_x - s / 2.0, center_y + s / 2.0]
        ]
        if as_tensor:
            base_square = torch.tensor(
                base_square, dtype=torch.float32, device=self.device
                ).unsqueeze(0).expand(B, -1, -1)
        
        return base_square

    def _compute_homography_from_points(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the homography matrix from 4 corresponding points via direct DLT (without normalization).
        src: [4, 2] source points.
        dst: [4, 2] destination points.
        Returns H: [3, 3] with H[2,2]=1.
        """
        A_rows = []
        b_rows = []
        for i in range(4):
            x, y = src[i]
            x_dst, y_dst = dst[i]
            row1 = torch.tensor([x, y, 1, 0, 0, 0, -x * x_dst, -y * x_dst],
                                  dtype=torch.float32, device=self.device)
            row2 = torch.tensor([0, 0, 0, x, y, 1, -x * y_dst, -y * y_dst],
                                  dtype=torch.float32, device=self.device)
            A_rows.append(row1)
            A_rows.append(row2)
            b_rows.append(torch.tensor(x_dst, dtype=torch.float32, device=self.device))
            b_rows.append(torch.tensor(y_dst, dtype=torch.float32, device=self.device))
        A = torch.stack(A_rows)  # [8, 8]
        b_vec = torch.stack(b_rows)  # [8]
        reg = 1e-4
        I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
        A_reg = A + reg * I
        # Solve for h vector
        h = torch.linalg.solve(A_reg, b_vec)
        H = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
        H[0, :3] = h[0:3]
        H[1, :3] = h[3:6]
        H[2, :2] = h[6:8]
        H[2, 2] = 1.0
        return H

    def solve_homography(self, predicted_square: torch.Tensor) -> torch.Tensor:
        """
        Compute homography H for each element in the batch using the predicted square.
        predicted_square: [B, 8] from CNN output (reshaped to [B, 4, 2]).
        Returns H: [B, 3, 3].
        """
        B = predicted_square.size(0)
        dst = predicted_square.view(B, 4, 2)
        base_square = self._create_base_square((B, 3, 0, 0))  # Shape will be fixed for each batch element.
        # We ignore image dimensions here since base_square is computed directly from predicted square dimensions.
        H_list = []
        for b in range(B):
            H_b = self._compute_homography_from_points(base_square[b], dst[b])
            H_list.append(H_b)
        return torch.stack(H_list, dim=0)

    def apply_homography(self, H: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Apply homography transformation.
        H: [B, 3, 3]
        points: [B, N, 2]
        Returns transformed points: [B, N, 2].
        """
        B, N, _ = points.shape
        ones = torch.ones((B, N, 1), device=points.device)
        points_h = torch.cat([points, ones], dim=2)
        points_transformed = torch.bmm(points_h, H.transpose(1, 2))
        points_transformed = points_transformed[:, :, :2] / points_transformed[:, :, 2:].clamp(min=1e-8)
        return points_transformed

    def extract_detection_points(self, detections: List[Detection], max_points: int = 30) -> torch.Tensor:
        """
        Extract foot position points (center-bottom of each box) from YOLO detections.
        Returns a tensor of shape [B, max_points, 2].
        """
        batch_points = []
        for det in detections:
            boxes = det.boxes  # [N, 4]
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
        return torch.stack(batch_points, dim=0)

    def _train_on_homography(self, images: torch.Tensor, gt_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training step for homography supervision.
        images: [B, C, H, W]
        gt_H: [B, 3, 3] ground truth homographies.
        Returns loss, predicted_square and target_square.
        """
        B = images.size(0)
        base_square = self._create_base_square(images.shape)
        target_square = self.apply_homography(gt_H, base_square)
        pred = self.cnn(images)  # Output shape: [B, 8]
        predicted_square = pred.view(B, 4, 2)
        loss = F.mse_loss(predicted_square, target_square)
        return loss, predicted_square, target_square

    def _train_on_player_positions(self, images: torch.Tensor, labels: torch.Tensor, detections: List[Detection]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step for player positions.
        images: [B, C, H, W]
        labels: [B, N, 2] ground truth player positions (in field coordinates).
        detections: YOLO detection outputs.
        Returns loss and projected_points.
        """
        self.current_image_shape = images.shape
        detections_pts = self.extract_detection_points(detections, max_points=labels.shape[1])
        pred = self.cnn(images)
        H = self.solve_homography(pred)
        projected_points = self.apply_homography(H, detections_pts)
        loss = F.mse_loss(projected_points, labels)
        return loss, projected_points

    def train_batches(self, epochs: int = -1) -> Generator[Tuple[int, str, int, torch.Tensor, torch.Tensor, float], None, None]:
        """
        Generator that yields training batch results.
        It runs two phases:
          - Homography training phase (if enabled).
          - Player position training phase (if enabled).
        Yields: (epoch, phase, batch_idx, images, predictions, loss)
        """
        epochs = self.epochs if epochs == -1 else epochs
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            # Phase 1: Homography training
            if self.config.train_on_homography:
                batch_idx = 0
                for batch in self.homography_loader:
                    images, gt_H = batch
                    images = images.to(self.device)
                    gt_H = gt_H.to(self.device)
                    loss, pred_sq, target_sq = self._train_on_homography(images, gt_H)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.logger.info(f"Homography Batch {batch_idx} Loss: {loss.item():.4f}")
                    yield (epoch, "homography", batch_idx, images.cpu(), pred_sq.cpu(), loss.item())
                    batch_idx += 1
            # Phase 2: Player position training
            if self.config.train_on_player_position:
                batch_idx = 0
                for batch in self.batches:
                    images = batch.image.to(self.device)
                    labels = batch.label.to(self.device)
                    detections = self.object_detect.detect(images)
                    loss, projected_pts = self._train_on_player_positions(images, labels, detections)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.logger.info(f"[Epoch {epoch + 1} Player Batch {batch_idx}] Loss: {loss.item():.4f}")
                    yield (epoch, "player", batch_idx, images.cpu(), projected_pts.cpu(), loss.item())
                    batch_idx += 1
            if self.save_path:
                self.save_model()

    def train(self, epochs: int = 1) -> None:
        """
        Standard training method that iterates over train_batches without yielding.
        """
        for _ in self.train_batches(epochs):
            pass
        self.logger.info("Training completed.")

    def inference(self, max_batches: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs inference on the player position training dataset.
        Returns a list of (projected_points, labels) tuples.
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
                pred = self.cnn(images)
                H = self.solve_homography(pred)
                projected_pts = self.apply_homography(H, self.extract_detection_points(detections, max_points=labels.shape[1]))
                results.append((projected_pts.cpu(), labels.cpu()))
        return results

    def evaluate(self, max_batches: int = 10) -> None:
        """
        Evaluate model performance on the player positions dataset using MSE loss.
        """
        results = self.inference(max_batches=max_batches)
        total_loss = 0.0
        count = 0
        for i, (projected_pts, labels) in enumerate(results):
            loss = F.mse_loss(projected_pts, labels).item()
            total_loss += loss
            count += 1
            if i == 0:
                self.logger.info(f"First batch MSE: {loss:.4f}")
        avg_loss = total_loss / max(count, 1)
        self.logger.info(f"Average Evaluation Loss: {avg_loss:.4f}")

    def save_model(self) -> None:
        """
        Saves the trained CNN model.
        """
        torch.save(self.cnn.state_dict(), self.save_path)


if __name__ == "__main__":
    # Example usage:
    # Set the dataset folder for homography training (annotated frames with ground truth H)
    config = TrainConfig(
        dataset_folder=Path("data/annotated_homographies"),
        epochs=10,
        batch_size=8,
        lr=1e-4,
        device="cuda",
        save_path=Path("data/10--models/perspect_cnn.pth"),
        train_on_homography=True,
        train_on_player_position=False  # Set to True if player position labels are available
    )
    trainer = PerspectTrainer(config)
    # Train using the combined phases
    trainer.train(epochs=10)
    # Optionally, evaluate and save the model
    trainer.evaluate(max_batches=5)
    trainer.save_model()
