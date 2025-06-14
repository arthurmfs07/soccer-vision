import cv2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from torchvision import transforms
from dataclasses import dataclass
from typing import *
from torch.utils.data import Dataset, DataLoader


from src.logger import setup_logger
from model.add_coords import AddCoords
from src.model.perspect.handcraft.model import build_model, BasePerspectModel
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
            transforms.Resize((320, 320)),
            transforms.ColorJitter(0.2,0.2,0.15,0.05),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
            AddCoords(),
        ])

        self.warp_alpha = 0.05

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i: int):
        img_p = self.img_paths[i]
        lbl_p = self.label_paths[i]
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        raw   = np.loadtxt(str(lbl_p), dtype=np.float32)
        pts   = raw[5:].reshape(-1, 3)                       # (32,3)
        ordered = np.zeros_like(pts)
        for rf_idx, (x, y, v) in enumerate(pts):
            ordered[rf2my[rf_idx]] = (x, y, v)

        coords_pix = ordered[:, :2].copy()                   # (32,2) normalised
        coords_pix[:, 0] *= w
        coords_pix[:, 1] *= h
        vis = (ordered[:, 2] >= 1)                          # bool (32,)

        # jitter
        src    = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        jitter = self.warp_alpha * np.array([[w, h]], np.float32)
        dst    = src + np.random.uniform(-jitter, +jitter, src.shape).astype(np.float32)
        H_pix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        H_inv    = np.linalg.inv(H_pix)

        img = cv2.warpPerspective(img, H_inv, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

        coords_pix = cv2.perspectiveTransform(
            coords_pix.reshape(-1, 1, 2), H_inv
        ).reshape(-1, 2)

        # zoom
        zoom = float(np.random.uniform(1.2, 1.6))
        nw, nh = int(w / zoom), int(h / zoom)
        x0 = np.random.randint(0, w - nw + 1)
        y0 = np.random.randint(0, h - nh + 1)

        # crop
        img_crop = img[y0:y0 + nh, x0:x0 + nw]
        img = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        coords_pix[:, 0] = (coords_pix[:, 0] - x0) * (w / nw)
        coords_pix[:, 1] = (coords_pix[:, 1] - y0) * (h / nh)

        # flip
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            coords_pix[:, 0] = w - coords_pix[:, 0]

        inside = (
            (coords_pix[:, 0] >= 0) & (coords_pix[:, 0] < w) &
            (coords_pix[:, 1] >= 0) & (coords_pix[:, 1] < h)
        )
        vis = vis & inside
        coords_pix[:, 0] = np.clip(coords_pix[:, 0], 0, w - 1)
        coords_pix[:, 1] = np.clip(coords_pix[:, 1], 0, h - 1)

        coords_norm = coords_pix.copy()
        coords_norm[:, 0] /= w
        coords_norm[:, 1] /= h

        img_t    = self.transform(img)                       # colour jitters etc.
        coords_t = torch.from_numpy(coords_norm).float()     # (32,2) in [0,1]
        vis_t    = torch.from_numpy(vis.astype(np.float32))  # (32,)

        return img_t, coords_t, vis_t




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
        self.logger = setup_logger("important-runs/2901_heatmap_resnet18_coords_entropy0.1.log")
        self.device = cfg.device

        self.sigma_pix:          float = 2.0   # std of target gauss in pixel
        self.excl_thr:           float = 0.0   # only penalize where actually expect point
        self.beta_excl_max:      float = 0.1   # weight of entropy term
        self.beta_warmup_epochs: int   = 50    # warmup for entropy term

        full_train = PointsDataset(cfg.dataset_folder, "train")
        full_val   = PointsDataset(cfg.dataset_folder, "valid")
        self.train_loader = DataLoader(full_train, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=4)
        self.val_loader   = DataLoader(full_val,   batch_size=cfg.batch_size,
                                       shuffle=False, num_workers=4)

        self.model = build_model(model_type="points", device=self.device)
        self.model.to(self.device)

        backbone_params = []
        head_params     = []
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if n.startswith("head"):
                    head_params.append(p)
                else:
                    backbone_params.append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr * 0.2, "weight_decay": 1e-4},  # lower lr for backbone
            {"params": head_params,     "lr": cfg.lr,       "weight_decay": 1e-4},
        ])

        # # train **all** parameters (cnn + head)
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=cfg.lr
        # )

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
            mode='min', factor=0.5, patience=cfg.patience, threshold=1e-4
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


    def train_batches(self):
        for b_idx, (imgs, coords_gt, vis_gt) in enumerate(self.train_loader):
            imgs, coords_gt, vis_gt = (imgs.to(self.device),
                                    coords_gt.to(self.device),
                                    vis_gt.to(self.device))
            
            # forward pass
            hmap_pred       = self.model(imgs)             # [B,32,H,W]
            B, K, Hm, Wm    = hmap_pred.shape

            # prepare values
            coords_pred, vis_log = self.model.heatmap_to_pts(hmap_pred)
            flat_pred       = hmap_pred.view(B,K,-1)
            logp_pred       = F.log_softmax(flat_pred, dim=-1)
            P               = torch.exp(logp_pred)

            # monitor peak
            with torch.no_grad():
                self.peak_sum   += P.max(-1)[0].mean().item()
                self.peak_count += 1

            # build gaussian targets
            sigma           = self.sigma_pix / max(Hm, Wm)
            hmap_tgt        = self.make_gaussian_heatmaps(coords_gt, vis_gt, Hm, Wm, sigma)
            flat_tgt        = hmap_tgt.view(B,K,-1)
            
            # KL + BCE losses
            kl_map          = F.kl_div(logp_pred, flat_tgt, reduction='none')
            ce_per_pt       = kl_map.sum(dim=-1)
            ce              = (ce_per_pt * vis_gt).sum() / (vis_gt.sum() + 1e-8)
            Lvis            = F.binary_cross_entropy_with_logits(vis_log, vis_gt)
            
            # ownership entropy regularization loss
            prob_sum        = P.sum(dim=1, keepdim=True)
            pi              = P / (prob_sum + 1e-8)
            entropy         = -(pi * (pi + 1e-8).log()).sum(dim=1)
            mask_true       = (flat_tgt.sum(dim=1) > self.excl_thr)
            L_excl          = (entropy * mask_true).sum() / (mask_true.sum() + 1e-8)

            train_loss_real = ce + Lvis + self.beta_excl * L_excl

            self.optimizer.zero_grad()
            train_loss_real.backward()
            self.optimizer.step()

            with torch.no_grad():
                mask    = vis_gt.unsqueeze(-1)
                Lpos    = F.mse_loss(coords_pred * mask, coords_gt * mask)
                Lvis2   = F.binary_cross_entropy_with_logits(vis_log, vis_gt)
                train_loss_metric = (Lpos + Lvis2).item()

            pred = torch.cat([coords_pred, vis_log.unsqueeze(-1)], dim=-1)  # [B,32,3]
            tgt  = torch.cat([coords_gt,    vis_gt.unsqueeze(-1)],    dim=-1)

            yield "points", b_idx, imgs[:,:3].cpu(), pred.cpu(), tgt.cpu(), train_loss_metric


    def train(self, epochs: int = -1, on_batch: Optional[Callable]=None):
        best, wait, e = float("inf"), 0, self.start_epoch
        max_e = epochs if epochs>0 else float("inf")

        while e < max_e:
            e+=1
            frac = min(e, self.beta_warmup_epochs) / self.beta_warmup_epochs
            self.peak_sum = 0.0
            self.peak_count = 0
            self.beta_excl = frac * self.beta_excl_max
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
                    # fmap      = self.model.cnn.conv_layers(imgs)
                    fmap      = self.model.conv_layers(imgs)
                    hmap_pred = self.model.head(fmap)

                    vis_gt    = vis_gt.to(self.device)

                    _, coords_pred, vis_log = self.model.predict(imgs)
                    mask  = vis_gt.unsqueeze(-1)
                    Lpos  = F.mse_loss(coords_pred * mask, coords_gt * mask)
                    Lvis  = F.binary_cross_entropy_with_logits(vis_log, vis_gt)
                    val_loss += (Lpos + Lvis).item()

                avg_val = val_loss / len(self.val_loader)
                self.avg_peak_prob = self.peak_sum / max(1, self.peak_count)
                self.logger.info(f"  val loss:      {avg_val:.4f}")
                self.logger.info(f"  avg_peak_prob: {self.avg_peak_prob:.4f}")
                self.lr_scheduler.step(avg_val)

                if avg_val < self.best_val:
                    self.best_val, wait = avg_val, 0
                    if self.cfg.save_path:
                        self.model.save_checkpoint(
                            self.cfg.save_path,
                            optimizer_state   = self.optimizer.state_dict(),
                            scheduler_state   = self.lr_scheduler.state_dict(),
                            best_val          = self.best_val,
                            epoch             = e,
                        )
                        self.logger.info(f"    saved best to {self.cfg.save_path}")
                else:
                    wait+=1
                    if wait>=self.cfg.patience:
                        self.logger.info("Early stopping.")
                        break

        self.logger.info("Finished training.")


    def make_gaussian_heatmaps(
            self,
            coords: torch.Tensor,   # [B,32,2] in [0..1]
            vis:    torch.Tensor,   # [B,32]  bool / {0,1}
            Hm: int, Wm: int,
            sigma_pix: float = 4.0,                 # σ **in pixels**
    ) -> torch.Tensor:                               # [B,32,Hm,Wm]
        B, K, _ = coords.shape
        device  = coords.device

        # grid of (x,y) in [0..1]   shape -> [1,1,Hm,Wm,2]
        ys, xs  = torch.linspace(0,1,Hm, device=device), torch.linspace(0,1,Wm, device=device)
        yy, xx  = torch.meshgrid(ys, xs, indexing='ij')
        grid    = torch.stack((xx, yy), dim=-1)[None, None]         # [1,1,Hm,Wm,2]

        tgt   = coords[:, :, None, None]                            # [B,32,1,1,2]
        d2    = (grid - tgt).square().sum(-1)                       # [B,32,Hm,Wm]

        var   = (sigma_pix / max(Hm, Wm))**2
        hmap  = torch.exp(-0.5 * d2 / var)

        hmap  = hmap * vis[:, :, None, None].float()

        denom = hmap.sum(dim=(-2, -1), keepdim=True)
        hmap  = hmap / (denom + 1e-8)

        return hmap
