import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Optional, Tuple, Generator, Callable
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader

from src.logger import setup_logger
from src.model.detect.objdetect import ObjectDetector
from src.model.perspect.cnn import CNN
from src.struct.detection import Detection
from src.utils import get_actual_yolo, get_csv_path, get_data_path
from src.model.perspect.batch import Batch
from src.visual.field import PitchConfig
from src.struct.utils import create_base_square

@dataclass
class TrainConfig:
    dataset_folder:           Path
    train_on_homography:      bool  = True
    train_on_player_position: bool  = False
    default_epochs:           int   = -1
    patience:                 int   = 5
    batch_size:               int   = 16
    lr:                       float = 1e-4
    warp_alpha:               float = 0.05
    device:                   str   = "cuda"
    save_path:       Optional[Path] = None


class HomographyDataset(Dataset):
    """
    Dataset for homography training.
    Expects files named as "annot_frame_<index>.png" and
    "annot_frame_<index>_H.npy" in the dataset_folder.
    Since the CNN is now adaptive, images are returned in full resolution.
    Returns (img_tensor, H_gt_tensor, frame_idx) so you can track the original index.
    """
    def __init__(self, dataset_folder: Path, warp_alpha: float = 0.05) -> None:
        self.dataset_folder = dataset_folder
        self.warp_alpha = warp_alpha
        self.pairs: List[Tuple[int, Path, Path]] = []

        # find all image files and their matching H files
        for img_path in sorted(dataset_folder.glob("annot_frame_*.png")):
            base = img_path.stem  # e.g., "annot_frame_42"
            parts = base.split("_")
            if len(parts) != 3:
                continue
            frame_idx = int(parts[2])
            h_path = dataset_folder / f"annot_frame_{frame_idx}_H.npy"
            if h_path.exists():
                self.pairs.append((frame_idx, img_path, h_path))


        self.transform = transforms.Compose([
            transforms.ToPILImage(),            # convert HxWxC ndarray -> PIL
            transforms.ColorJitter(             # random color/brightness
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),                             
            transforms.RandomGrayscale(p=0.05), # occasionally drop color
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0)), # simulate blur
            transforms.ToTensor()               # back to [0..1] float tensor
        ])

        if not self.pairs:
            raise ValueError(f"No valid homography pairs found in {dataset_folder}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        frame_idx, img_path, h_path = self.pairs[i]
        I = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H_orig = np.load(str(h_path))

        h, w = I.shape[:2]
        src = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
        jitter = self.warp_alpha * np.array([[w,h]])
        dst = src + np.random.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
        W, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        W_inv = np.linalg.inv(W)
        I_warp = cv2.warpPerspective(
            I, W_inv, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        H_warp = H_orig @ W_inv

        # with 50% chance, flip horizontally
        flip = np.random.rand() < 0.5
        if flip:
            w_pr = PitchConfig().length
            I_warp = cv2.flip(I_warp, 1)
            F_pr = np.array([
                [   -1,    0, w_pr],
                [    0,    1,    0],
                [    0,    0,    1]
            ], dtype=np.float32)
            H_warp = F_pr @ H_warp

        aff_norm = np.linalg.norm(H_warp[:2,:2])
        H_warp  /= max(aff_norm, 1e-4)
        img_tensor = self.transform(I_warp)

        return (
            img_tensor, 
            torch.from_numpy(H_warp).float(), 
            frame_idx,
            flip
        )


class PerspectTrainer:
    """
    Trainer for two-stage training of a perspective homography CNN.
    The first stage (if enabled) overfits on annotated homography pairs.
    The second stage (if enabled) takes YOLO detections, applies the estimated homography,
    and compares the projected player positions with ground truth labels.
    """
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.logger = setup_logger("train.log")
        self.epochs = config.default_epochs
        self.batch_size = config.batch_size
        self.save_path = config.save_path
        self.device = config.device

        self.data_dir = get_data_path()
        self.csv_path = get_csv_path()
        self.yolo_path = get_actual_yolo()

        self.tpl_w = PitchConfig().length
        self.tpl_h = PitchConfig().width

        # player positions loader
        self.batches = Batch(self.data_dir, self.csv_path, batch_size=self.batch_size)
        self.object_detect = ObjectDetector(model_path=self.yolo_path,
                                           device=self.device)

        # homography loader
        self.homography_dataset = HomographyDataset(self.config.dataset_folder, self.config.warp_alpha)
        self.homography_loader = DataLoader(
            self.homography_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        # model and optimizer
        self.cnn = CNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.config.lr)
        self.current_image_shape: Optional[Tuple[int]] = None

    def hungarian_loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Hungarian-matched MSE loss.
        yhat: [B, N, 2] predicted positions.
        y:    [B, N, 2] ground truth positions.
        """
        B, N, _ = yhat.shape
        total_loss = 0.0
        for b in range(B):
            pred = yhat[b]
            target = y[b]
            pred = pred[(pred != -1).all(dim=1)]
            target = target[(target != -1).all(dim=1)]
            if pred.size(0) == 0 or target.size(0) == 0:
                continue
            cost = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            total_loss += F.mse_loss(pred[row_ind], target[col_ind])
        return total_loss / B

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
            A_rows.append(torch.tensor([x, y, 1, 0, 0, 0, -x * x_dst, -y * x_dst],
                                       dtype=torch.float32, device=self.device))
            A_rows.append(torch.tensor([0, 0, 0, x, y, 1, -x * y_dst, -y * y_dst],
                                       dtype=torch.float32, device=self.device))
            b_rows.append(torch.tensor(x_dst, dtype=torch.float32, device=self.device))
            b_rows.append(torch.tensor(y_dst, dtype=torch.float32, device=self.device))
        A = torch.stack(A_rows)
        b_vec = torch.stack(b_rows)
        reg = 1e-4
        A_reg = A + reg * torch.eye(A.size(-1), device=A.device)
        h = torch.linalg.solve(A_reg, b_vec)
        H = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
        H[0, :3] = h[0:3]
        H[1, :3] = h[3:6]
        H[2, :2] = h[6:8]
        H[2, 2] = 1.0
        return H

    def solve_homography(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute homography H for each element in the batch using the predicted square.
        predicted_square: [B, 8] from CNN output (reshaped to [B, 4, 2]).
        Returns H: [B, 3, 3].
        """
        B, _, _, _ = self.current_image_shape
        corners = pred_norm.view(B, 4, 2)
        corners_px = corners.clone()
        corners_px[..., 0] *= self.tpl_w
        corners_px[..., 1] *= self.tpl_h

        base_square = create_base_square((B, 3, 0, 0), as_tensor=True)
        H_list = []
        for b in range(B):
            H_list.append(self._compute_homography_from_points(base_square[b], corners_px[b]))
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
        pts_h = torch.cat([points, ones], dim=2)
        trans = torch.bmm(pts_h, H.transpose(1, 2))
        return trans[:, :, :2] / trans[:, :, 2:].clamp(min=1e-8)

    def extract_detection_points(self, detections: List[Detection], max_points: int = 30) -> torch.Tensor:
        """
        Extract foot position points (center-bottom of each box) from YOLO detections.
        Returns a tensor of shape [B, max_points, 2].
        """
        batch_pts = []
        for det in detections:
            pts = []
            for x1, y1, x2, y2 in det.boxes:
                pts.append([(x1 + x2) / 2, y2])
            if pts:
                pts_tensor = torch.tensor(pts, dtype=torch.float32, device=self.device)
            else:
                pts_tensor = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            num = pts_tensor.size(0)
            if num < max_points:
                pad = torch.full((max_points - num, 2), -1, device=self.device)
                pts_tensor = torch.cat([pts_tensor, pad], dim=0)
            else:
                pts_tensor = pts_tensor[:max_points]
            batch_pts.append(pts_tensor)
        return torch.stack(batch_pts, dim=0)

    def _train_on_homography(
        self, images: torch.Tensor, gt_H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training step for homography supervision.
        images: [B, C, H, W]
        gt_H: [B, 3, 3] ground truth homographies.
        Returns (loss, pred_field, target_norm).
        """
        B = images.size(0)
        base_sq = create_base_square(images.shape, as_tensor=True)
        target_px = self.apply_homography(gt_H, base_sq)
        target_norm = target_px.clone()
        target_norm[..., 0] /= self.tpl_w
        target_norm[..., 1] /= self.tpl_h

        pred_norm = self.cnn(images)
        loss = F.mse_loss(pred_norm, target_norm)

        pred_field = pred_norm.clone()
        pred_field[..., 0] *= self.tpl_w
        pred_field[..., 1] *= self.tpl_h
        return loss, pred_field, target_norm

    def _train_on_player_positions(
        self, images: torch.Tensor, labels: torch.Tensor, detections: List[Detection]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step for player positions.
        """
        self.current_image_shape = images.shape
        det_pts = self.extract_detection_points(detections, max_points=labels.shape[1])
        pred = self.cnn(images)
        H = self.solve_homography(pred)
        proj_pts = self.apply_homography(H, det_pts)
        loss = F.mse_loss(proj_pts, labels)
        return loss, proj_pts

    def train_batches(self) -> Generator[Tuple[int, str, int, torch.Tensor, torch.Tensor, float], None, None]:
        """
        Generator yielding (epoch, phase, batch_idx, images, predictions, loss).
        Spots any sample with error > 1e3 and logs its frame index.
        """

        if self.config.train_on_homography:
            for batch_idx, (images, gt_H, frame_idxs, flips) in enumerate(self.homography_loader):
                B = images.size(0)
                images = images.to(self.device)
                gt_H = gt_H.to(self.device)

                # compute target_norm
                base_sq = create_base_square(images.shape, as_tensor=True)
                target_px = self.apply_homography(gt_H, base_sq)

                if flips.any():
                    swap_lr = torch.tensor([1,0,3,2],device=target_px.device)
                    target_px[flips] = target_px[flips][:, swap_lr]

                target_norm = target_px.clone()
                target_norm[..., 0] /= self.tpl_w
                target_norm[..., 1] /= self.tpl_h

                # forward + loss
                pred_norm = self.cnn(images)
                loss = F.mse_loss(pred_norm, target_norm)

                # per-sample error check
                with torch.no_grad():
                    errs = (pred_norm - target_norm).view(B, -1).pow(2).sum(dim=1)
                    for i, e in enumerate(errs):
                        if e.item() > 1e3:
                            bad_frame = frame_idxs[i].item()
                            self.logger.warning(
                                f"💥 exploding loss on frame {bad_frame}: err={e.item():.2e}"
                            )

                # backward + step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), 1.0)
                self.optimizer.step()
                self.logger.info(f"    Batch {batch_idx}: Loss: {loss.item():.4f}")

                # yield for visualization
                pred_field = pred_norm.clone()
                pred_field[..., 0] *= self.tpl_w
                pred_field[..., 1] *= self.tpl_h
                yield ("homography", batch_idx,
                        images.cpu(), 
                        pred_field.cpu(),
                        target_px.cpu(), 
                        loss.item())

        if self.config.train_on_player_position:
            for batch_idx, batch in enumerate(self.batches):
                images = batch.image.to(self.device)
                labels = batch.label.to(self.device)
                detections = self.object_detect.detect(images)
                loss, proj_pts = self._train_on_player_positions(images, labels, detections)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                yield ("player", batch_idx,
                        images.cpu(), 
                        proj_pts.cpu(), 
                        target_px.cpu(),
                        loss.item())


    def train(self, epochs: int = -1, on_batch: Optional[Callable] = None) -> None:
        """
        Early-stopping training loop that calls train_batches() epoch-by-epoch
        """
        if epochs < 0 and self.config.patience < 0:
            raise ValueError("If epochs = -1 (infinite training), patience must be > 0.")

        max_epochs = float('inf') if epochs < 0 else epochs
        best_loss = float("inf")
        wait = 0
        epoch = 0

        while epoch < max_epochs:
            epoch += 1
            self.logger.info(f"Epoch {epoch}/{max_epochs}")

            epoch_loss, n_batches = 0.0, 0    

            for phase, batch_idx, images, preds, gt_pts, loss in self.train_batches():
                if on_batch:
                    on_batch(phase, batch_idx, images, preds, gt_pts, loss)
                if phase == "homography":
                    epoch_loss += loss
                    n_batches  += 1
                
            if n_batches == 0:
                self.logger.warning(f"No homography batches in epoch {epoch}; skipping.")
                break

            avg_loss = epoch_loss / n_batches
            self.logger.info(f"Epoch {epoch}/{max_epochs} - avg homography loss: {avg_loss:.4f}")

            # Early-stop check
            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
                if self.config.save_path:
                    self.save_model()
            else:
                wait += 1
                if wait >= self.config.patience:
                    self.logger.info(f"No improvement for {wait} epoch (patience={self.config.patience}); stopping early.")
                    break

        self.logger.info("Training Finished.")


    def inference(self, max_batches: int = 10):
        """
        Runs inference on the player position dataset.
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
                proj_pts = self.apply_homography(H, self.extract_detection_points(detections, max_points=labels.shape[1]))
                
                results.append((proj_pts.cpu(), labels.cpu()))
        return results

    def evaluate(self, max_batches: int = 10) -> None:
        """
        Evaluate model performance on the player positions dataset using MSE loss.
        """
        results = self.inference(max_batches=max_batches)
        total_loss = 0.0
        for i, (proj_pts, labels) in enumerate(results):
            loss = F.mse_loss(proj_pts, labels).item()
            total_loss += loss
            if i == 0:
                self.logger.info(f"First batch MSE: {loss:.4f}")
        avg_loss = total_loss / max(1, len(results))
        self.logger.info(f"Average Evaluation Loss: {avg_loss:.4f}")

    def save_model(self) -> None:
        """
        Saves the trained CNN model.
        """
        torch.save(self.cnn.state_dict(), self.save_path)


if __name__ == "__main__":
    config = TrainConfig(
        dataset_folder=Path("data/annotated_homographies"),
        batch_size=8,
        lr=1e-4,
        device="cuda",
        save_path=Path("data/10--models/perspect_cnn.pth"),
        train_on_homography=True,
        train_on_player_position=False
    )
    trainer = PerspectTrainer(config)
    trainer.train(epochs=10)
    trainer.evaluate(max_batches=5)
    trainer.save_model()
