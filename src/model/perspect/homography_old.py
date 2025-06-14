import cv2
import torch
import numpy as np
import warnings
from typing import *
from src.logger import setup_logger
from dataclasses import dataclass

from src.visual.field import FieldVisualizer, PitchConfig
from src.config import HEConfig


class HomographyEstimator:
    def __init__(
            self,
            device: str = "cuda"
            ):
        cfg = HEConfig()
        self.cfg = cfg
        self.device = device

        self.smooth_alpha = cfg.smooth_alpha
        self.max_err      = cfg.max_err
        self.max_pts      = cfg.max_pts 
        self.min_pts      = cfg.min_pts
        self.min_inl      = cfg.min_inl
        self.det_lo       = cfg.det_lo
        self.det_hi       = cfg.det_hi
        self.thr_px       = cfg.thr_px

        fv = FieldVisualizer(PitchConfig())
        self.canonical_points = fv._reference_model_pts()

        self._prev_H = np.eye(3, dtype=np.float32)
        self._prev_log_Hs: Optional[List[np.ndarray]] = None

        self.logger = setup_logger("homography.log")

    def predict(
            self,
            coords: torch.Tensor, 
            vis_prob: torch.Tensor,
            conf_thr: float = 0.5
            ) -> List[np.ndarray]:
        self.conf_thr = conf_thr
        Hs_raw = self._fit_homography_from_points(coords, vis_prob)
        Hs_int = self._fill_missing_homographies(Hs_raw)
        Hs     = self._smooth_homographies(Hs_int)

        return Hs
    


    def _fit_homography_from_points(
            self,
            coords:      torch.Tensor,   # [B,32,2], normalized [0–1]
            vis_probs:   torch.Tensor,   # [B,32]
        ) -> List[np.ndarray]:

        import logging
        self.logger.setLevel(logging.DEBUG)

        B            = coords.size(0)
        canon        = self.canonical_points  # (32,2) in [0,1]
        coords_np    = coords.cpu().numpy().astype(np.float32)
        probs_np     = vis_probs.cpu().numpy()
        vis_idx      = [np.where(p >= self.conf_thr)[0] for p in probs_np]
        H_list: List[Optional[np.ndarray]] = [None] * B

        for b in range(B):
            idx = vis_idx[b]
            self.logger.debug(f"[{b}] pts={idx.size}")
            if idx.size < self.min_pts: continue

            idx  = idx[np.argsort(-probs_np[b,idx])[:self.max_pts]]
            src  = coords_np[b,idx]
            dst  = canon[idx]
            try:
                H, mask = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.thr_px,
                    confidence=0.999,
                    maxIters=500
                )
            except cv2.error:
                continue

            if H is None or mask is None or mask.sum() < self.min_inl:
                continue
        
            H /= H[2,2] + 1e-12
            d = abs(np.linalg.det(H))
            if not self.det_lo < d < self.det_hi:
                self.logger.debug("    → det(H)≈0 — fallback")
                continue

            try:
                err = _msre(H, src, dst)
            except np.linalg.LinAlgError:    
                self.logger.debug("    → inv(H) failed — fallback")
                continue
 
            self.logger.debug(f"  msre: {err:.4f}")
            if not math.isfinite(err) or err > self.max_err:
                self.logger.debug("    → msre too large")
                continue
                       
            self.logger.debug("    → ACCEPTED")
            H_list[b] = H.astype(np.float32)
            self._prev_H = H_list[b]

        return H_list


    def _fill_missing_homographies(
            self,
            Hs: List[Optional[np.ndarray]]
            ) -> torch.Tensor:
        """
        Lie‐algebra interpolation as before.
        """
        idx = [i for i,H in enumerate(Hs) if H is not None]
        if not idx:
            return torch.from_numpy(np.stack([self._prev_H]*len(Hs))).to(self.device)

        L = {}
        for i in idx:
            Li = _safe_logm(Hs[i])
            if Li is None:
                Li = _safe_logm(self._prev_H)
            if Li is None:
                Li = np.zeros((3,3), np.float32)
            L[i] = Li

        out = []
        for i in range(len(Hs)):
            if Hs[i] is not None:
                out.append(Hs[i])
                continue

            l = max([j for j in idx if j < i], default=None)
            r = min([j for j in idx if j > i], default=None)

            if l is None:
                out.append(Hs[r])
            elif r is None:
                out.append(Hs[l])
            else:
                t = (i-l)/(r-l)
                Hi = expm((1-t)*L[l] + t*L[r]).real.astype(np.float32)
                Hi /= Hi[2,2]
                out.append(Hi)
        return torch.from_numpy(np.stack(out)).to(self.device)


    def _smooth_homographies(
            self, 
            Hs_t: torch.Tensor
        ) -> torch.Tensor:
        """
        Exponential smoothing in Lie algebra; Hs_t is (B,3,3) on self.device.
        """
        Hs = Hs_t.cpu().numpy()
        B  = Hs.shape[0]

        if self._prev_log_Hs is None or len(self._prev_log_Hs) != B:
            self._prev_log_Hs = [None] * B

        out, new_logs = [], []
        for i in range(B):
            with np.errstate(all="ignore"):

                L_now = _safe_logm(Hs[i])
                if L_now is None:
                    L_now = self._prev_log_Hs[i]
                if L_now is None:
                    L_now = np.zeros((3,3), np.float32)

            L_prev = self._prev_log_Hs[i]
            L_s    = L_now if L_prev is None else self.smooth_alpha*L_prev + (1-self.smooth_alpha)*L_now
            H_s    = expm(L_s).real.astype(np.float32)
            H_s   /= H_s[2,2]
            out.append(H_s)
            new_logs.append(L_s)

        self._prev_log_Hs = new_logs
        self._prev_H      = out[-1]
        return torch.from_numpy(np.stack(out)).to(self.device)



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


def _safe_logm(H: np.ndarray, thr: float=1e-6) -> Optional[np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        L, err = logm(H, disp=False)
    if err > thr or not np.isfinite(L).all():
        return None
    return np.real(L).astype(np.float32)