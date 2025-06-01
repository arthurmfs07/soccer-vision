import cv2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Optional, Tuple, Generator, Callable
from torch.utils.data import Dataset, DataLoader, random_split

from src.logger import setup_logger
from src.model.detect.objdetect import ObjectDetector
from src.data.add_coords import AddCoords
from src.model.perspect.model import build_model, BasePerspectModel
from src.utils import get_actual_yolo, get_csv_path, get_data_path
from src.model.perspect.batch import Batch
from src.struct.utils import create_base_square


@dataclass
class SquareTrainConfig:
    dataset_folder:           str
    default_epochs:           int   = -1
    patience:                 int   = 5
    batch_size:               int   = 16
    lr:                       float = 1e-4
    warp_alpha:               float = 0.05
    device:                   str   = "cuda"
    area_edge_weight:         float = 0.1
    save_path:       Optional[str]  = None



class HomographyDataset(Dataset):
    """
    Dataset for homography training.
    Expects files named `frame_XXXXXX.png` & `frame_XXXXXX_H.npy`.
    Returns (img_tensor, H_gt, frame_idx, flip_flag).
    """
    def __init__(
            self, 
            dataset_folder: str, 
            warp_alpha:     float = 0.05
            ) -> None:
        self.dataset_folder = Path(dataset_folder)
        self.warp_alpha = warp_alpha

        self.pairs: List[Tuple[int, Path, Path]] = []
        for img_path in sorted(self.dataset_folder.glob("frame_*.png")):
            idx = int(img_path.stem.split("_")[-1])
            h_path = self.dataset_folder / f"{img_path.stem}_H.npy"
            if h_path.exists():
                self.pairs.append((idx, img_path, h_path))

        if not self.pairs:
            raise ValueError(f"No valid homography pairs in {dataset_folder}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.05),
            transforms.RandomGrayscale(0.05),
            # transforms.GaussianBlur(5,(0.1,2.0)),
            transforms.ToTensor(),
            AddCoords(),
        ])

    def __len__(self) -> int:
        return len(self.pairs)


    def __getitem__(self, i: int):
        idx, img_path, H_path = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H_gt_norm = np.load(str(H_path)).astype(np.float32)  # [3×3]

        h, w = img.shape[:2]

        src = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
        jitter = self.warp_alpha * np.array([[w,h]])
        dst    = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
        H_pix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        img = cv2.warpPerspective(img, np.linalg.inv(H_pix), (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)


        zoom = float(np.random.uniform(1.0, 1.4))
        nw   = int(w / zoom); nh = int(h / zoom)
        x0   = np.random.randint(0, w - nw + 1)
        y0   = np.random.randint(0, h - nh + 1)
        cropped = img[y0:y0+nh, x0:x0+nw]
        img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        H_pix_inv = np.linalg.inv(H_pix)
        T_norm    = np.diag([1.0/w, 1.0/h, 1.0]).astype(np.float32)
        T_denorm  = np.diag([w, h, 1.0]).astype(np.float32)
        H_pix_inv_norm = T_norm @ H_pix_inv @ T_denorm

        H_gt_norm = H_gt_norm @ H_pix_inv_norm

        M_zoom = np.array([
            [1/zoom,     0, x0],
            [    0, 1/zoom, y0],
            [    0,     0,   1],
        ], dtype=np.float32)
        Mz_norm = T_norm @ M_zoom @ T_denorm
        H_gt_norm = H_gt_norm @ Mz_norm

        flip = (np.random.rand() < 0.5)
        if flip:
            img     = cv2.flip(img, 1)
            F_pr    = np.array([[-1,0,1],[0,1,0],[0,0,1]], dtype=np.float32)
            H_gt_norm = F_pr @ H_gt_norm

        H_gt_norm /= H_gt_norm[2,2].item()

        img_t = self.transform(img)
        H_t   = torch.from_numpy(H_gt_norm).float()
        return img_t, H_t, idx, flip



class PerspectSquareTrainer:
    """
    Two‐stage trainer for perspective homography (and optional player‐position).
    Delegates all homography + warping logic to PerspectModel.
    """
    def __init__(self, cfg: SquareTrainConfig) -> None:
        self.cfg      = cfg
        self.logger   = setup_logger("train.log")
        self.device   = cfg.device

        # detection & player batches (optional)
        self.batches  = Batch(get_data_path(), get_csv_path(),
                              batch_size=cfg.batch_size)
        self.detector = ObjectDetector(get_actual_yolo(), device=self.device)

        full_ds = HomographyDataset(cfg.dataset_folder, cfg.warp_alpha)
        n_val = int(len(full_ds) * 0.15)
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            shuffle=True, num_workers=4
        )
        self.val_loader   = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            shuffle=False, num_workers=4
        )

        # PerspectModel
        self.model = build_model(model_type="square", device=self.device)
        self.model.to(self.device)
        
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.cfg.patience,
            threshold=1e-4
        )
        

        if cfg.save_path and Path(cfg.save_path).is_file():
            self.model, meta = BasePerspectModel.load_checkpoint(
                cfg.save_path,
                device=cfg.device,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
            )
            self.best_val    = meta.get("best_val", float("inf"))
            self.start_epoch = meta.get("epoch",    0)
            print(f"→ resumed from epoch {self.start_epoch}, best_val={self.best_val:.4f}")
        else:
            self.best_val, self.start_epoch = float("inf"), 0

    
    def _polygon_area(self, pts: torch.Tensor) -> torch.Tensor:
        """Shoelace formula on [B,4,2] -> [B]"""
        x, y = pts.unbind(-1)
        # roll so edges (0->1, 1->2, 2-> 3, 3->0)
        x2, y2 = x.roll(shifts=-1, dims=1), y.roll(shifts=-1, dims=1)
        return 0.5 * torch.abs((x * y2 - x2 * y).sum(dim=-1))

    def _edge_lengths(self, pts: torch.Tensor) -> torch.Tensor:
        """Compute 3 edge lengths for each quad -> [B,4]"""
        diffs = pts - pts.roll(shifts=-1, dims=1) # [B,4,2]
        return diffs.norm(dim=-1)      # [B,4]

    def _homography_loss(
            self, 
            images: torch.Tensor, 
            H_gt: torch.Tensor,
            flips: torch.Tensor):
        B = images.size(0)
        
        pred_sq = self.model(images)        # [B,4,2]
        base_sq = create_base_square(as_tensor=True)       # [1,4,2]
        base_sq = base_sq.to(self.device).expand(B, -1, -1)
        target_sq = self.model.warp_points(H_gt, base_sq)  # [B,4,2]

        if flips.any():
            swap = torch.tensor([1,0,3,2], device=target_sq.device)
            flips_idx = flips.nonzero(as_tuple=True)[0]
            target_sq[flips_idx] = target_sq[flips_idx][:, swap]

        loss_mse = F.mse_loss(pred_sq, target_sq)

        # edge + area regularizer
        pred_e = self._edge_lengths(pred_sq)
        tft_e  = self._edge_lengths(target_sq)
        L_edge = F.l1_loss(pred_e, tft_e)

        pred_a = self._polygon_area(pred_sq)
        tgt_a  = self._polygon_area(target_sq)
        L_area = F.l1_loss(pred_a, tgt_a)

        loss = loss_mse + self.cfg.area_edge_weight * (L_edge + L_area)
        return loss, pred_sq, target_sq


    def _extract_detection_points(self, dets, max_pts):
        batch_pts = []
        for det in dets:
            pts = [((x1+x2)/2, y2) for x1,y1,x2,y2 in det.boxes]
            t = torch.tensor(pts, device=self.device) \
                if pts else torch.empty((0,2),device=self.device)
            if t.size(0)<max_pts:
                pad = torch.full((max_pts-t.size(0),2), -1.,device=self.device)
                t = torch.cat([t,pad],dim=0)
            batch_pts.append(t[:max_pts])
        return torch.stack(batch_pts,dim=0)
    
    def train_batches(self) -> Generator:
        # Homography stage
        for b_idx, (imgs, Hs, f_idxs, flips) in \
                enumerate(self.train_loader):
            imgs, Hs, flips = (imgs.to(self.device), 
                                Hs.to(self.device), 
                                flips.to(self.device))
            loss, pred, target = self._homography_loss(imgs, Hs, flips)

            # debug ranges
            tmin, tmax = target.min().item(), target.max().item()
            pmin, pmax = pred.min().item(),   pred.max().item()
            self.logger.debug(
                f"[H] target ∊ [{tmin:.3f},{tmax:.3f}], pred ∊ [{pmin:.3f},{pmax:.3f}]"
            )

            # backward + step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            yield "homography", b_idx, imgs.cpu(), pred.cpu(), target.cpu(), loss.item()


    def train(self, epochs: int = -1, on_batch: Optional[Callable] = None):
        """test method for standalone"""
        best_val, wait, e = float("inf"), 0, 0
        max_e = float("inf") if epochs<0 else epochs

        while e < max_e:
            e += 1
            self.logger.info(f"Epoch {e}/{max_e}")

            self.model.train()
            train_loss, n_train = 0.0, 0
            for tpl in self.train_batches():
                phase, idx, imgs, preds, targets, loss_val = tpl
                if on_batch: on_batch(*tpl)
                train_loss += loss_val; n_train+=1
            avg_train_loss = train_loss / max(1, n_train)

            self.model.eval()
            val_loss, n_val = 0., 0
            with torch.no_grad():
                for imgs, Hs, _, flips in self.val_loader:
                    imgs, Hs, flips = (imgs.to(self.device), 
                                       Hs.to(self.device),
                                       flips.to(self.device))
                    loss_val = self._homography_loss(imgs, Hs, flips)[0].item()
                    val_loss += loss_val
                    n_val+=1
                avg_val_loss = val_loss / max(1, n_val)
                self.logger.info(f"    val loss: {avg_val_loss:.4f}")
                self.lr_scheduler.step(avg_val_loss)

                if avg_val_loss < self.best_val:
                    self.best_val, wait = avg_val_loss, 0
                    if self.cfg.save_path:
                        self.model.save_checkpoint(
                            self.cfg.save_path,
                            optimizer_state   = self.optimizer.state_dict(),
                            scheduler_state   = self.lr_scheduler.state_dict(),
                            best_val          = self.best_val,
                            epoch             = e,
                        )
                        self.logger.info(f"    new best model, saved to {self.cfg.save_path}")
                else:
                    wait += 1
                    self.logger.info(f"    patience: {wait}/{self.cfg.patience}")
                    if wait >= self.cfg.patience:
                        self.logger.info("    patience exhausted; early stop.")
                        break
        self.logger.info("Finished training.")


    def save_model(self):
        self.model.save(self.cfg.save_path)



if __name__ == "__main__":

    dataset_folder = "data/01--clean/roboflow"
        
    config = SquareTrainConfig(
        dataset_folder=dataset_folder,
        batch_size=8,
        lr=1e-4,
        device="cuda",
        save_path="data/10--models/perspect_cnn.pth",
        train_on_homography=True,
        train_on_player_position=False
    )


    trainer = PerspectSquareTrainer(config)
    trainer.train(epochs=10)
    trainer.evaluate(max_batches=5)
    trainer.save_model()
