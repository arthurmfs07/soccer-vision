
# YOLO:
# Observation -> detection

# CNN:
# Observation -> Homography entries

# CNN Loss:
# detection @ homography vs real frame position

import torch
from torch.nn import functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any

from src.logger import setup_logger
from src.model.detect.objdetect import ObjectDetector
from src.model.perspect.cnn import CNN
from src.struct.detection import Detection
from src.utils import get_actual_yolo, get_csv_path, get_data_path
from src.model.perspect.batch import Batch, BatchData
from src.visual.field import PitchConfig

@dataclass
class TrainConfig:
    save_path: Path = None # field(default_factory=load_abs_path() / "perspect_cnn.pth")
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
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


    def loss_fn(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between homography-transformed detections and ground truth."""
        mask = targets != -1
        mask = mask.all(dim=2) # [B, N]
        masked_preds = predictions[mask]
        masked_targets = targets[mask]
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        return F.mse_loss(masked_preds, masked_targets)
        

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

                detections = self.object_detect.detect(images)
                detections_pts = self.extract_detection_points(detections, max_points=labels.shape[1])
                
                h_flat = self.cnn(images)
                H = h_flat.view(-1, 3, 3)

                yhat_norm = self.apply_homography(H, detections_pts)
                labels_norm = self._field_norm(labels)

                yhat_field_model = self._field_denorm(yhat_norm)

                loss = self.loss_fn(yhat_norm, labels_norm)
                
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


    def train(self, epochs: int =1):
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

                h_flat = self.cnn(images)
                H = h_flat.view(-1, 3, 3)

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
