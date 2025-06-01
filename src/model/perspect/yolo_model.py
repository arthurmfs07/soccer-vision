import torch
from ultralytics import YOLO
from typing import *
import numpy as np

from src.logger import setup_logger
from src.struct.utils              import create_base_square
from src.visual.field             import FieldVisualizer, PitchConfig
from src.utils import rf2my


class YOLOModel:
    """
    Wraps a YOLOv8-pose model so that

        Hs, coords, vis = model.predict(X, predict_H=True)

    returns:
      • Hs:    Tensor[B,3,3]    — identity homographies  
      • coords:Tensor[B,K,2]    — (x,y) of each of K keypoints  
      • vis:   Tensor[B,K]      — probability [0..1] of each keypoint  
    """

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        imgsz: int = 640,
        iou: float  = 0.45,
    ):
        self.device = device
        self.imgsz  = imgsz
        # load and send to device
        self._yolo = YOLO(weights)
        self._yolo.to(device)
        # set inference defaults
        self._yolo.overrides = dict(conf=0.9, iou=iou)

        self.logger = setup_logger("yolo_model.log")

        # read K from the model yaml
        self.K = int(self._yolo.model.yaml["kpt_shape"][0])

        fv = FieldVisualizer(PitchConfig())
        self.canonical_points = fv._reference_model_pts()
        
        self._prev_H = np.eye(3, dtype=np.float32)


    def eval(self):
        # no‐op so InferenceProcess.model.eval() works
        pass

    def predict(
        self,
        X: torch.Tensor,
        predict_H: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          X:          Tensor[B,3,H,W], values in [0..1] or [0..255]
          predict_H:  whether to return identity homographies (ignored internally)
        Returns:
          Hs:    Tensor[B,3,3]
          coords:Tensor[B,K,2]
          vis:   Tensor[B,K]
        """
        B = X.shape[0]
        # run YOLOv8-pose inference (batch‐wise)
        results = self._yolo.predict(
            source=X,
            device=self.device,
            imgsz=self.imgsz,
            augment=False,
            verbose=False,
        )

        coords_list = []
        vis_list    = []

        for r in results:
            # r.keypoints: Tensor[N, K, 3] or None
            if r.keypoints is None or r.keypoints.shape[0] == 0:
                # no detections → all zeros
                coords_list.append(torch.zeros((self.K, 2), device=self.device))
                vis_list.append(torch.zeros((self.K,   ), device=self.device))
            else:
                xy = r.keypoints.xyn
                conf = r.keypoints.conf

                first_xy = xy[0]
                first_conf = conf[0]
                coords_list.append(first_xy.to(self.device))
                vis_list.append(first_conf.to(self.device))


        K = len(coords_list)
        inv = [0]*K
        for old_i, new_i in rf2my.items():
            inv[new_i] = old_i

        for i in range(len(coords_list)):
            coords_list[i] = coords_list[i][inv, :]
            vis_list[i]    = vis_list[i][inv]

        # stack into batched tensors
        coords   = torch.stack(coords_list, dim=0)  # (B,K,2)
        vis_prob = torch.stack(vis_list,    dim=0)  # (B,K)

        # build homographies
        if not predict_H:
            return None, coords, vis_prob
        H = self._fit_homography_from_points(coords, vis_prob, conf_th=self.conf_th)

        return H, coords, vis_prob

    __call__ = predict


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


