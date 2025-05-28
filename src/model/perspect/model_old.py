import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *
from src.model.perspect.cnn import CNN

class PerspectModel(nn.Module):
    """
    Dual-mode perspective homography model.
    Modes:
      - 'square': predict 4 corner points -> homography
      - 'points': predict N reference points + visibility -> homography via RANSAC
    """

    def __init__(
            self, 
            train_type: Literal["square", "points"] = "square",
            device: str = "cuda"
            ):
        super().__init__()
        assert train_type in ("square", "points"), "train_type must be 'square' or 'points'"
        self.train_type = train_type
        self.device     = device

        self.heatmap_size: Tuple[int,int] = (40, 40)
        
        self.cnn = CNN().to(device)
        # self.cnn = ResNet().to(device)
        
        if train_type == "square":
            out_dim = 8
            layers = nn.Sequential(
                nn.Linear(self.cnn.feature_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, out_dim)
            ).to(device)

        elif train_type == "points":
            in_ch = self.cnn.config.hidden_channels[-1]
            out_ch = 32
            layers = [
                nn.ConvTranspose2d(in_ch, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, out_ch, kernel_size=1), # [B,32,4Hf,4Wf]
                nn.Upsample(size=self.heatmap_size,
                            mode='bilinear', align_corners=False)
            ]

            self.register_buffer('pixel_coords', torch.empty(0))

        self.head = nn.Sequential(*layers).to(device)
        self._init_head()

        self.canonical_points = self._get_canonical_points()


    def forward(
            self, 
            X: torch.Tensor
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          - if square mode: (pts4, None) where pts4 is [B,4,2] in [0..1]
          - if points mode: (coords, vis_log) where
                coords   is [B,32,2] in [0..1]
                vis_log  is [B,32]   in logits for visibility
        """

        if self.train_type == "square":
            feats = self.cnn(X)      # [B, feature_dim]
            raw   = self.head(feats) # [B, 8] or [B,32*3]
            pts4 = torch.sigmoid(raw).view(-1, 4, 2)
            return pts4
        
        elif self.train_type == "points":
            fmap = self.cnn.conv_layers(X)
            hmap = self.head(fmap)
            return hmap


    def predict(
            self, 
            X: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        forward + solve. 
        Returns:
        - H as [B,3,3]
        - coords
        - vis_log
        """
        y = self.forward(X)

        if self.train_type == "square":
            coords = y
            H = self._fit_homography_from_square(coords)
        elif self.train_type == "points":

            B, N, Hm, Wm = y.shape

            flat  = y.view(B, N, -1)
            probs = F.softmax(flat, dim=-1) # [B,32,P]

            if self.pixel_coords.numel() == 0:
                ys = torch.linspace(0, 1, Hm, device=X.device)
                xs = torch.linspace(0, 1, Wm, device=X.device)
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')
                grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
                self.pixel_coords = grid
            
            coords = probs.matmul(self.pixel_coords) # [B,32,2]
            conf_logits = flat.max(dim=-1).values    # [B,32]
        
            H = self._fit_homography_from_points(coords, conf_logits)
        return (H, coords, conf_logits)


    def _fit_homography_from_square(
        self,
        square: torch.Tensor
    ) -> torch.Tensor:
        """
        square: [B,4,2] in normalized pixel coordinates.
        solves homography
        """
        src     = self.canonical_points # np.array [4,2]
        Hs      = []
        for dst in square.detach().cpu().numpy().astype(np.float32):
            H  = cv2.getPerspectiveTransform(src, dst)
            Hs.append((H / H[2,2]).astype(np.float32))
        return torch.from_numpy(np.stack(Hs, 0)).to(self.device)


    def _fit_homography_from_points(
            self,
            coords: torch.Tensor,
            vis_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        coords:   [B,32,2]
        vis_mask: [B,32] boolean
        """
        canon     = self.canonical_points # [32,2]
        Hs        = []
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
        mask_np   = vis_mask.detach().cpu().numpy()
        
        for b in range(coords_np.shape[0]):
            m = mask_np[b]
            dst = coords_np[b, m]
            src = canon[m]

            H = None
            corner_idxs = self._select_extreme_corners(src)
            if corner_idxs is not None:
                src_pts = src[corner_idxs]
                dst_pts = dst[corner_idxs]
                try:
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                except cv2.error:
                    H = None

            if H is None:
                try:
                    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                except cv2.error:
                    H = None
            
            if H is None:
                H = Hs[-1] if Hs else np.eye(3, dtype=np.float32)

            Hs.append((H / H[2,2]).astype(np.float32))
        return torch.from_numpy(np.stack(Hs, 0)).to(self.device)

    def _select_extreme_corners(self, pts: np.ndarray) -> Optional[List[int]]:
        """
        Given (N,2) array of points, pick up to 4 indices 
        corresponding to most extreme points in each angular gradant around the centroid.
        Return list of 4 indices or None if there aren't enough.
        """
        centroid = pts.mean(axis=0)
        vecs = pts - centroid
        dist = np.linalg.norm(vecs, axis=1)
        ang = (np.arctan2(vecs[:,1], vecs[:,0]) + 2*np.pi) % (2*np.pi)
        
        idxs = []
        for q in range(4):
            lo, hi = q * np.pi/2, (q+1)*np.pi/2
            mask = (ang >= lo) & (ang < hi)
            if mask.any():
                candidates = np.where(mask)[0]
                farthest    = candidates[np.argmax(dist[mask])]
                idxs.append(int(farthest))
        
        unique = []
        for i in idxs:
            if i not in unique:
                unique.append(i)
        if len(unique) < 4:
            for i in np.argsort(-dist):
                if i not in unique:
                    unique.append(int(i))
                if len(unique) == 4:
                    break
        
        return unique if len(unique) == 4 else None



    def warp_points(
            self,
            H: torch.Tensor,
            pts: torch.Tensor
        ) -> torch.Tensor:
        """
        H:   [B,3,3]
        pts: [B,N,2] in normalized video‐coords
        → returns [B,N,2] in normalized field‐coords
        """
        H_np   = H.detach().cpu().numpy()
        pts_np = pts.detach().cpu().numpy()
        out    = []
        for b in range(pts_np.shape[0]):
            src = pts_np[b].reshape(-1,1,2).astype(np.float32)
            dst = cv2.perspectiveTransform(src, H_np[b]).reshape(-1,2)
            out.append(dst)
        return torch.from_numpy(np.stack(out,0).astype(np.float32)).to(self.device)

    def warp_from_image(self,
                        images: torch.Tensor,
                        pts: torch.Tensor) -> torch.Tensor:
        """
        convenience: H = predict_homography(images); return warp_points(H, pts)
        """
        H    = self.predict_homography(images)
        return self.warp_points(H, pts)

    def save_checkpoint(
        self,
        path: str,
        *,
        optimizer_state: Optional[Dict[str,Any]]   = None,
        scheduler_state: Optional[Dict[str,Any]]   = None,
        best_val:        Optional[float]           = None,
        epoch:           Optional[int]             = None
    ):
        """
        Save a full training checkpoint at `path`, including:
          - model_state
          - optimizer_state (if given)
          - lr_scheduler_state (if given)
          - best_val (if given)
          - epoch (if given)
        """
        ckpt: Dict[str,Any] = {
            "model_state": self.state_dict(),
            "train_type": self.train_type
        }
        if optimizer_state is not None:
            ckpt["optimizer_state"]   = optimizer_state
        if scheduler_state is not None:
            ckpt["lr_scheduler_state"] = scheduler_state
        if best_val is not None:
            ckpt["best_val"] = best_val
        if epoch is not None:
            ckpt["epoch"] = epoch

        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: str,
        *,
        optimizer:   Optional[torch.optim.Optimizer]    = None,
        scheduler:   Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Load a checkpoint dict from `path`.  Returns:
          model:        an instance of PerspectModel (with weights loaded)
          metadata:     dict with keys 'best_val', 'epoch', etc.
        If you pass in optimizer and/or scheduler, their state_dicts will also be loaded.
        """
        ckpt = torch.load(path, map_location=device)
        # instantiate same model type:
        model = cls(train_type=ckpt.get("train_type","square"), device=device)
        model.load_state_dict(ckpt["model_state"])

        metadata: Dict[str,Any] = {}
        for key in ("best_val","epoch"):
            if key in ckpt:
                metadata[key] = ckpt[key]

        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "lr_scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["lr_scheduler_state"])

        return model, metadata


    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)


    def _get_canonical_points(self) -> np.ndarray:
        """
        - square: return 4x2 array of image square
        - points: return 32x2 array of reference points
        """

        if self.train_type == "square":
            from src.struct.utils import create_base_square
            return create_base_square(as_tensor=False).astype(np.float32)
        elif self.train_type == "points":
            from src.visual.field import FieldVisualizer, PitchConfig
            fv = FieldVisualizer(PitchConfig())
            return fv._reference_model_pts().astype(np.float32) # return np.ndarray [32,2]


    def predict_square(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,H,W] in [0..1]
        returns: [B,4,2] predicted corners in normalized field‐space
        """
        return self.cnn(images)

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
            
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


