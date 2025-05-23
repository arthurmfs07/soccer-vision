import cv2
import torch
import torch.nn as nn
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
        
        self.cnn = CNN().to(device)
        # self.cnn = ResNet().to(device)
        out_dim = 8 if train_type == "square" else 32 * 3

        self.head = nn.Sequential(
            nn.Linear(self.cnn.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        ).to(device)
        
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
        feats = self.cnn(X)      # [B, feature_dim]
        raw   = self.head(feats) # [B, 8] or [B,32*3]

        if self.train_type == "square":
            pts4 = torch.sigmoid(raw).view(-1, 4, 2)
            return pts4, None
        
        pts_vis = raw.view(-1, 32, 3)
        coords  = torch.sigmoid(pts_vis[..., :2]) # x,y
        vis_log = pts_vis[..., 2]  # visibility logits
        return coords, vis_log

    def predict_homography(self, X: torch.Tensor) -> torch.Tensor:
        """
        forward + solve. Returns H as [B,3,3]
        """
        coords, vis_log = self.forward(X)

        if self.train_type == "square":
            return self._fit_homography_from_square(coords)
        
        vis_mask = vis_log > 0.5
        return self._fit_homography_from_points(coords, vis_mask)
    

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
        canon = self.canonical_points # [32,2]
        Hs = []
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
        mask_np   = vis_mask.detach().cpu().numpy()
        for b in range(coords_np.shape[0]):
            m = mask_np[b]
            dst = coords_np[b, m]
            src = canon[m]
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
            Hs.append((H / H[2,2]).astype(np.float32))
        return torch.from_numpy(np.stack(Hs, 0)).to(self.device)


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
                torch.nn.init.uniform_(m.weight, a=-1e-3, b=1e-3)
                torch.nn.init.constant_(m.bias, 0.0)


