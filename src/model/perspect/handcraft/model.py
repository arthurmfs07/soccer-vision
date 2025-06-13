import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as tvm
from typing import *

from src.logger import setup_logger
from src.model.perspect.handcraft.cnn import CNN
from src.struct.utils              import create_base_square
from src.visual.field             import FieldVisualizer, PitchConfig


def build_model(
    model_type: Literal["square","points"] = "square",
    device:     str                        = "cuda",
    heatmap_size: Tuple[int,int]           = (40,40)
) -> nn.Module:
    """
    Factory: returns a SquarePerspectModel or PointPerspectModel.
    """
    if model_type == "square":
        return SquarePerspectModel(device=device)
    elif model_type == "points":
        return PointPerspectModel(device=device, heatmap_size=heatmap_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def build_resnet(
    *,
    arch: str = "resnet18",   # "resnet18","resnet34","resnet50",…
    pretrained: bool = True,
    num_extra: int = 2,       # 0 → vanilla RGB
    trainable_layers: Sequence[str] = ("layer4", "fc"),
    replace_stride_with_dilation: Tuple[bool, bool, bool] = (False, False, False),
) -> Tuple[nn.Module, int]:
    """
    return (backbone, feature_dim)
    """

    weights = getattr(tvm, f"{arch[0].upper()}{arch[1:]}_Weights", None)
    net = getattr(tvm, arch)(
        weights = weights.DEFAULT if pretrained and weights else None,
        replace_stride_with_dilation = replace_stride_with_dilation
    )

    if num_extra > 0:
        old = net.conv1     # Conv2d(3,64,7,2,3)
        new = nn.Conv2d(
            in_channels  = old.in_channels + num_extra,  # 3+N
            out_channels = old.out_channels,
            kernel_size  = old.kernel_size,
            stride       = old.stride,
            padding      = old.padding,
            bias         = False
        )
        new.weight.data[:, :3] = old.weight.data
        nn.init.kaiming_normal_(new.weight.data[:, 3:], mode="fan_out", nonlinearity="relu")
        net.conv1 = new

    for name, param in net.named_parameters():
        param.requires_grad = any(name.startswith(tl) for tl in trainable_layers)

    feat_dim = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }[arch]

    net.fc = nn.Identity()
    return net, feat_dim


class BasePerspectModel(nn.Module):
    """
    Shared backbone, homography solvers, warp helpers, checkpoint I/O.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        # self.cnn    = CNN().to(device)

        self.cnn, self.feature_dim = build_resnet(
            arch="resnet18",
            pretrained=True,
            num_extra=2,  # add 2 extra channels for coordinates
            trainable_layers=("conv1","bn1","layer1","layer2","layer3","layer4","fc"),
        )

        self.cnn.to(device)
        self.cnn.eval()
        self.feature_dim = 512 # = self.cnn.feature_dim
        self.conv_layers = nn.Sequential(
            self.cnn.conv1, self.cnn.bn1, 
            self.cnn.relu, self.cnn.maxpool,
            self.cnn.layer1, self.cnn.layer2, 
            self.cnn.layer3, self.cnn.layer4
        )

        self.logger = setup_logger("model.log")

    def forward(self, X: torch.Tensor):
        raise NotImplementedError

    def predict(self, X: torch.Tensor):
        raise NotImplementedError

    def warp_points(
        self,
        H:   torch.Tensor,
        pts: torch.Tensor
    ) -> torch.Tensor:
        H_np   = H.detach().cpu().numpy()
        pts_np = pts.detach().cpu().numpy()
        out    = []
        for b in range(pts_np.shape[0]):
            src = pts_np[b].reshape(-1,1,2).astype(np.float32)
            dst = cv2.perspectiveTransform(src, H_np[b]).reshape(-1,2)
            out.append(dst)
        return torch.from_numpy(np.stack(out,0).astype(np.float32)).to(self.device)

    def warp_from_image(
        self,
        images: torch.Tensor,
        pts:    torch.Tensor
    ) -> torch.Tensor:
        H, coords, conf = self.predict(images)
        return self.warp_points(H, pts)

    def save_checkpoint(
        self,
        path:            str,
        *,
        optimizer_state: Optional[Dict[str,Any]] = None,
        scheduler_state: Optional[Dict[str,Any]] = None,
        best_val:        Optional[float]         = None,
        epoch:           Optional[int]           = None
    ):
        ckpt: Dict[str,Any] = {
            "model_state": self.state_dict()
        }
        # subclasses may define self.train_type
        if hasattr(self, "train_type"):
            ckpt["train_type"] = self.train_type
        if optimizer_state   is not None: ckpt["optimizer_state"]   = optimizer_state
        if scheduler_state   is not None: ckpt["lr_scheduler_state"] = scheduler_state
        if best_val          is not None: ckpt["best_val"]           = best_val
        if epoch             is not None: ckpt["epoch"]              = epoch

        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(
        cls,
        path:      str,
        device:    str,
        *,
        optimizer: Optional[torch.optim.Optimizer]            = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        ckpt = torch.load(path, map_location=device)
        mtype = ckpt.get("train_type", "square")
        model = build_model(mtype, device=device)
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

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)



class SquarePerspectModel(BasePerspectModel):
    """Predict 4 corners directly, then solve homography."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.train_type = "square"

        # MLP head for 4 corners
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 8)
        ).to(self.device)
        self._init_head()

        self.canonical_points = create_base_square(as_tensor=False).astype(np.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(X)           # [B, feature_dim]
        raw   = self.head(feats)      # [B,8]
        return torch.sigmoid(raw).view(-1,4,2) # [B,4,2]

    def predict(
        self,
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pts4 = self.forward(X)           # [B,4,2]
        H    = self._fit_homography_from_square(pts4)
        return H, pts4, None


    def _fit_homography_from_square(self, square: torch.Tensor) -> torch.Tensor:
        src = self.canonical_points  # np.ndarray [4,2]
        Hs  = []
        for dst in square.detach().cpu().numpy().astype(np.float32):
            H     = cv2.getPerspectiveTransform(src, dst)
            Hs.append((H / (H[2,2] + 1e-12)).astype(np.float32))
        return torch.from_numpy(np.stack(Hs,0)).to(self.device)


class PointPerspectModel(BasePerspectModel):
    """Predict 32 heat-maps + soft-argmax, then solve homography."""

    def __init__(
        self,
        device:       str              = "cuda",
        heatmap_size: Tuple[int,int]  = (40,40)
    ):
        super().__init__(device)
        self.train_type   = "points"
        self.heatmap_size = heatmap_size
        self._prev_H = np.eye(3, dtype=np.float32)

        self.conf_th: float = 0.15

        # self.feature_dim = self.cnn.config.hidden_channels[-1]

        in_ch, out_ch = (
            self.feature_dim,
            32
        )
        # buffer for soft-arg-max grid
        self.register_buffer('pixel_coords', torch.empty(0), persistent=False)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1),
            # nn.Upsample(size=self.heatmap_size,
            #             mode='bilinear',
            #             align_corners=False),
        ).to(self.device)
        self._init_head()

        # canonical 32 reference points
        fv = FieldVisualizer(PitchConfig())
        self.canonical_points = fv._reference_model_pts().astype(np.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # fmap = self.cnn.conv_layers(X)  # [B, Cf, Hf, Wf]
        fmap = self.conv_layers(X)  # [B, Cf, Hf, Wf]
        hmap = self.head(fmap)          # [B,32,Hm,Wm]
        return hmap


    def heatmap_to_pts(
        self,
        hmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw heat-maps -> (coords [B,32,2], conf_logits [B,32])
        via spatial softmax + soft-arg-max.
        """
        B, N, Hm, Wm = hmap.shape
        flat        = hmap.view(B, N, -1)            # [B,32,P]
        probs       = F.softmax(flat, dim=-1)        # [B,32,P]

        if self.pixel_coords.numel() == 0:
            ys      = torch.linspace(0,1,Hm, device=hmap.device)
            xs      = torch.linspace(0,1,Wm, device=hmap.device)
            yy, xx  = torch.meshgrid(ys, xs, indexing='ij')
            self.pixel_coords = torch.stack(
                [xx.flatten(), yy.flatten()], dim=1
            )  # [P,2]

        coords      = probs.matmul(self.pixel_coords)      # [B,32,2]
        conf_logits = flat.max(dim=-1).values              # [B,32]
        return coords, conf_logits


    def predict(
        self,
        X: torch.Tensor,
        predict_H: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hmap = self.forward(X)              # [B,32,Hm,Wm]
        coords, vis_log = self.heatmap_to_pts(hmap) # [B,32,2], [B,32]

        if not predict_H:
            return None, coords, vis_log
        vis_prob = torch.sigmoid(vis_log)  # [B,32]
        H = self._fit_homography_from_points(coords, vis_prob, conf_th=self.conf_th)
        return H, coords, vis_prob



    def _fit_homography_from_points(
            self,
            coords:      torch.Tensor,   # [B,32,2], normalized [0–1]
            vis_probs:   torch.Tensor,   # [B,32]
            conf_th:     float = 0.45,   # min confidence
            max_pts:     int   = 8,      # PROSAC‐style cap
            max_err:     float = 0.07     # MSRE threshold in normalized units
        ) -> torch.Tensor:

        import logging
        self.logger.setLevel(logging.DEBUG)

        thr = 2.0 / 320

        device, B    = coords.device, coords.shape[0]
        canon        = self.canonical_points.astype(np.float32)  # (32,2) in [0,1]
        coords_np    = coords.cpu().numpy().astype(np.float32)
        probs_np     = vis_probs.cpu().numpy()

        from src.model.perspect.homography_qc import HomographyQC
        qc = HomographyQC(interactive=True)

        H_list: List[Optional[np.ndarray]] = []
        for b in range(B):
            inds = np.where(probs_np[b] >= conf_th)[0]
            self.logger.debug(f"\n–– frame {b} ––")
            self.logger.debug(f"  #pts ≥ th: {inds.size}")
            if inds.size < 4:
                self.logger.debug(f"    → too few points {inds.size}")
                H_list.append(self._prev_H.copy())
                continue

            inds = inds[np.argsort(-probs_np[b,inds])[:max_pts]]
            src  = canon[inds]
            dst  = coords_np[b,inds]
            try:
                H, mask = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=thr,
                    confidence=0.999,
                    maxIters=500
                )
            except cv2.error:
                self.logger.debug("    → cv2.findHomography failed")
                H = None

            if H is None or mask is None or mask.sum() < 4:
                H_list.append(self._prev_H.copy())
                continue
        
            if abs(H[2,2]) < 1e-8:
                self.logger.debug("    → H[2,2] too small (singular) — fallback")
                H_list.append(self._prev_H.copy())
                continue

            H = (H / (H[2,2] + 1e-12)).astype(np.float32)

            if abs(np.linalg.det(H)) < 1e-6:
                self.logger.debug("    → det(H)≈0 — fallback")
                H_list.append(self._prev_H.copy())
                continue

            try:
                err = _msre(H, src, dst)
            except np.linalg.LinAlgError:    
                self.logger.debug("    → inv(H) failed — fallback")
                H_list.append(self._prev_H.copy())
                continue
 
            self.logger.debug(f"  msre: {err:.4f}")
            if not math.isfinite(err) or err > max_err:
                self.logger.debug("    → msre too large")
                H_list.append(self._prev_H.copy())
                continue

            # qc.inspect(frame_index=b, H=H, src=src, dst=dst, err=err)
                       
            self.logger.debug("    → ACCEPTED")
            H_list.append(H.copy())
            self._prev_H = H.copy()

        return torch.from_numpy(np.stack(H_list,0)).to(device)


    def _fill_missing_homographies(
            self,
            Hs: List[Optional[np.ndarray]]
            ) -> List[np.ndarray]:
        """
        Lie‐algebra interpolation as before.
        """
        idxs = [i for i,H in enumerate(Hs) if H is not None]
        if not idxs:
            H_def = getattr(self, "_prev_H", np.eye(3, dtype=np.float32))
            return [H_def.copy() for _ in Hs]

        logs = {}
        for i in idxs:
            try:
                logs[i] = logm(Hs[i])
            except LinAlgError:
                logs[i] = logm(self._prev_H)

        out = [None]*len(Hs)
        for i in range(len(Hs)):
            if Hs[i] is not None:
                out[i] = Hs[i]
                continue
            left  = max([j for j in idxs if j < i], default=None)
            right = min([j for j in idxs if j > i], default=None)

            if left is None:
                out[i] = Hs[right].copy()
            elif right is None:
                out[i] = Hs[left].copy()
            else:
                t   = (i - left)/(right - left)
                L   = (1-t)*logs[left] + t*logs[right]
                Hi  = expm(L).real.astype(np.float32)
                out[i] = Hi / Hi[2,2]
        return out



import cv2, numpy as np, math
from typing import List, Optional
from scipy.linalg import logm, expm, LinAlgError

def _msre(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """
    Mean symmetric reprojection error in *pixels* of the normalized system.
    Actually here both src and dst are in [0,1], so this is in normalized units.
    """
    src_h = cv2.perspectiveTransform(src[None], H)[0]
    dst_h = cv2.perspectiveTransform(dst[None], np.linalg.inv(H))[0]
    err_f = np.linalg.norm(src_h - dst, axis=1).mean()
    err_b = np.linalg.norm(dst_h - src, axis=1).mean()
    return 0.5 * (err_f + err_b)


