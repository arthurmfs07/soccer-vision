import cv2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from torchvision import transforms
from dataclasses import dataclass
from typing import *
from torch.utils.data import Dataset, DataLoader, random_split

from src.logger import setup_logger
from src.model.perspect.model import PerspectModel
from src.utils import rf2my  # your roboflow→ours index map

@dataclass
class PointsTrainConfig:
    dataset_folder:           str
    default_epochs:           int   = -1
    patience:                 int   = 5
    batch_size:               int   = 16
    lr:                       float = 1e-4
    device:                   str   = "cuda"
    save_path:       Optional[str]  = None

class PointsDataset(Dataset):
    """
    Loads images + Roboflow‐style labels:
      each .txt has 5 header floats (ignored), then 32×(x,y,vis)
    We remap raw‐idx→our‐idx via rf2my.
    Returns (img_tensor, coords_gt, vis_gt)
    """
    def __init__(self, root: str, split: str="train"):
        base    = Path(root) / split
        img_dir = base / "images"
        lbl_dir = base / "labels"

        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not imgs:
            raise ValueError(f"No images found in {img_dir}")

        self.img_paths   = []
        self.label_paths = []

        for img_p in imgs:
            stem   = img_p.stem
            lbl_p  = lbl_dir / f"{stem}.txt"
            if not lbl_p.exists():
                raise FileNotFoundError(f"Label file not found for {img_p}: expected {lbl_p}")
            self.img_paths.append(img_p)
            self.label_paths.append(lbl_p)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.2,0.2,0.15,0.05),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i:int):
        img_p = self.img_paths[i]
        lbl_p = self.label_paths[i]
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        img_t = self.transform(img)

        # load raw label: one line of floats
        raw = np.loadtxt(str(lbl_p), dtype=np.float32)
        # drop first 5
        pts = raw[5:].reshape(-1,3)   # shape (32,3)
        # remap
        ordered = np.zeros_like(pts)  # (32,3)
        for rf_idx,(x,y,v) in enumerate(pts):
            my_idx = rf2my[rf_idx]
            ordered[my_idx] = (x,y,v)
        coords = ordered[:,:2]        # (32,2)
        vis    = (ordered[:,2]>=1).astype(np.float32)  # (32,)
        return img_t, torch.from_numpy(coords), torch.from_numpy(vis)

class PerspectPointsTrainer:
    """
    Trainer for points‐based homography head.
    Almost identical to PerspectTrainer, but:
     - dataset = PointsDataset
     - model = PerspectModel(train_type="points")
     - loss = MSE(coords) + BCE(vis)
    """
    def __init__(self, cfg: PointsTrainConfig):
        self.cfg    = cfg
        self.logger = setup_logger("train_points.log")
        self.device = cfg.device

        full_train = PointsDataset(cfg.dataset_folder, "train")
        full_val   = PointsDataset(cfg.dataset_folder, "valid")
        self.train_loader = DataLoader(full_train, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=4)
        self.val_loader   = DataLoader(full_val,   batch_size=cfg.batch_size,
                                       shuffle=False, num_workers=4)

        self.model = PerspectModel(train_type="points", device=self.device)
        self.model.to(self.device)

        if cfg.save_path:
            ckpt = Path(cfg.save_path)
            if ckpt.is_file():
                print(f"→ loading checkpoint from {cfg.save_path}")
                self.model.load(cfg.save_path)


        # train **all** parameters (cnn + head)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr
        )
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
            mode='min', factor=0.5, patience=cfg.patience, threshold=1e-4
        )

    def train_batches(self) -> Generator:
        for b_idx, (imgs, coords_gt, vis_gt) in enumerate(self.train_loader):
            imgs      = imgs.to(self.device)
            coords_gt = coords_gt.to(self.device)
            vis_gt    = vis_gt.to(self.device)

            # forward + loss
            coords_pred, vis_log = self.model(imgs)   # [B,32,2], [B,32]
            # positional MSE (mask-out invisible points)
            mask = vis_gt.unsqueeze(-1)
            n_vis = mask.sum().clamp(min=1.0)
            Lpos = F.smooth_l1_loss(
                coords_pred * mask,
                coords_gt   * mask,
                reduction="sum"
            ) / n_vis
            Lvis = F.binary_cross_entropy_with_logits(vis_log, vis_gt)
            loss = Lpos + Lvis

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            pred = torch.cat([coords_pred, vis_log.unsqueeze(-1)], dim=-1)
            tgt  = torch.cat([coords_gt,    vis_gt.unsqueeze(-1)], dim=-1)

            yield "points", b_idx, imgs.cpu(), pred.cpu(), tgt.cpu(), loss.item()

    def train(self, epochs: int = -1, on_batch: Optional[Callable]=None):
        best, wait, e = float("inf"), 0, 0
        max_e = epochs if epochs>0 else float("inf")

        while e < max_e:
            e+=1
            self.logger.info(f"Epoch {e}/{max_e}")
            # train
            self.model.train()
            train_loss = 0.0
            for phase, idx, imgs, pred, tgt, loss in self.train_batches():
                if on_batch: on_batch(phase, idx, imgs, pred, tgt, loss)
                train_loss += loss
            self.logger.info(f"  train loss: {train_loss/len(self.train_loader):.4f}")

            # validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, coords_gt, vis_gt in self.val_loader:
                    imgs      = imgs.to(self.device)
                    coords_gt = coords_gt.to(self.device)
                    vis_gt    = vis_gt.to(self.device)

                    coords_pred, vis_log = self.model(imgs)
                    mask  = vis_gt.unsqueeze(-1)
                    Lpos  = F.mse_loss(coords_pred * mask, coords_gt * mask)
                    Lvis  = F.binary_cross_entropy_with_logits(vis_log, vis_gt)
                    val_loss += (Lpos+Lvis).item()

                avg_val = val_loss / len(self.val_loader)
                self.logger.info(f"  val loss:   {avg_val:.4f}")
                self.lr_scheduler.step(avg_val)

                if avg_val < best:
                    best, wait = avg_val, 0
                    if self.cfg.save_path:
                        self.model.save(self.cfg.save_path)
                        self.logger.info(f"    saved best to {self.cfg.save_path}")
                else:
                    wait+=1
                    if wait>=self.cfg.patience:
                        self.logger.info("Early stopping.")
                        break

        self.logger.info("Finished training.")

    def save_model(self):
        if self.cfg.save_path:
            self.model.save(self.cfg.save_path)
