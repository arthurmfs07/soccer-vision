import cv2
import torch
import numpy as np
import warnings
import math
from collections import deque
from typing import *
from dataclasses import dataclass
from scipy.linalg import logm, expm

from src.logger import setup_logger
from src.visual.field import FieldVisualizer, PitchConfig
from src.config import HEConfig


@dataclass
class FrameData:
    H: Optional[np.ndarray]
    L: Optional[np.ndarray]
    inliers: int
    msre: float
    valid: bool


class HomographyEstimator:
    def __init__(
            self,
            device: str = "cuda",
            half_window: int = 6,
            cut_threshold: float = 5.0
            ):
        cfg = HEConfig()
        self.cfg = cfg
        self.device = device

        self.K = half_window
        self.window_size = 2 * self.K + 1
        self.cut_threshold = cut_threshold
        
        self.max_err = cfg.max_err
        self.max_pts = cfg.max_pts 
        self.min_pts = cfg.min_pts
        self.min_inl = cfg.min_inl
        self.det_lo = cfg.det_lo
        self.det_hi = cfg.det_hi
        self.thr_px = cfg.thr_px

        fv = FieldVisualizer(PitchConfig())
        self.canonical_points = fv._reference_model_pts()

        self._cache: Deque[FrameData] = deque(maxlen=self.window_size)
        self._prev_H = np.eye(3, dtype=np.float32)
        self._output_buffer: List[np.ndarray] = []

        self.logger = setup_logger("homography.log")

    def predict(
            self,
            coords: torch.Tensor, 
            vis_prob: torch.Tensor,
            conf_thr: float = 0.5
            ) -> torch.Tensor:
        self.conf_thr = conf_thr
        B = coords.size(0)
        
        raw_data = self._fit_homography_from_points(coords, vis_prob)
        
        self._output_buffer.clear()
        
        for b in range(B):
            frame = raw_data[b]
            
            if self._detect_cut(frame):
                self._cache.clear()
            
            self._cache.append(frame)
            
            if len(self._cache) == self.window_size:
                H_smooth = self._smooth_center_frame()
                self._output_buffer.append(H_smooth)
        
        while len(self._output_buffer) < B:
            if len(self._cache) > 0:
                H_smooth = self._smooth_partial_window()
                self._output_buffer.append(H_smooth)
            else:
                self._output_buffer.append(self._prev_H)
        
        return torch.from_numpy(np.stack(self._output_buffer[:B])).to(self.device)

    def _fit_homography_from_points(
            self,
            coords: torch.Tensor,
            vis_probs: torch.Tensor,
        ) -> List[FrameData]:
        B = coords.size(0)
        canon = self.canonical_points
        coords_np = coords.cpu().numpy().astype(np.float32)
        probs_np = vis_probs.cpu().numpy()
        
        frames = []
        
        for b in range(B):
            idx = np.where(probs_np[b] >= self.conf_thr)[0]
            
            if idx.size < self.min_pts:
                frames.append(FrameData(None, None, 0, np.inf, False))
                continue
            
            idx = idx[np.argsort(-probs_np[b,idx])[:self.max_pts]]
            src = coords_np[b,idx]
            dst = canon[idx]
            
            try:
                H, mask = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.thr_px,
                    confidence=0.999,
                    maxIters=500
                )
            except cv2.error:
                frames.append(FrameData(None, None, 0, np.inf, False))
                continue
            
            if H is None or mask is None:
                frames.append(FrameData(None, None, 0, np.inf, False))
                continue
            
            inliers = int(mask.sum())
            if inliers < self.min_inl:
                frames.append(FrameData(None, None, inliers, np.inf, False))
                continue
            
            H = H.astype(np.float32)
            H /= H[2,2] + 1e-12
            
            d = abs(np.linalg.det(H))
            if not self.det_lo < d < self.det_hi:
                frames.append(FrameData(None, None, inliers, np.inf, False))
                continue
            
            try:
                err = _msre(H, src, dst)
            except np.linalg.LinAlgError:
                frames.append(FrameData(None, None, inliers, np.inf, False))
                continue
            
            if not math.isfinite(err) or err > self.max_err:
                frames.append(FrameData(None, None, inliers, np.inf, False))
                continue
            
            L = _safe_logm(H)
            if L is None:
                frames.append(FrameData(None, None, inliers, err, False))
                continue
            
            frames.append(FrameData(H, L, inliers, err, True))
            self._prev_H = H
        
        return frames

    def _interpolate_logs(self) -> None:
        n = len(self._cache)
        valid_idx = [i for i in range(n) if self._cache[i].valid]
        
        if not valid_idx:
            for i in range(n):
                L = _safe_logm(self._prev_H)
                self._cache[i].L = L if L is not None else np.zeros((3,3), np.float32)
            return
        
        for i in range(n):
            if self._cache[i].valid:
                continue
            
            left = [j for j in valid_idx if j < i]
            right = [j for j in valid_idx if j > i]
            
            if not left:
                self._cache[i].L = self._cache[right[0]].L.copy()
            elif not right:
                self._cache[i].L = self._cache[left[-1]].L.copy()
            else:
                l, r = left[-1], right[0]
                t = (i - l) / (r - l)
                self._cache[i].L = (1-t) * self._cache[l].L + t * self._cache[r].L

    def _smooth_center_frame(self) -> np.ndarray:
        self._interpolate_logs()
        
        weights = np.array([
            f.inliers / max(f.msre, 1e-6) if f.valid else 1e-3
            for f in self._cache
        ])
        
        if weights.sum() < 1e-9:
            weights[:] = 1.0
        
        weights /= weights.sum()
        
        L_mean = np.zeros((3,3), dtype=np.float32)
        for w, f in zip(weights, self._cache):
            if f.L is not None:
                L_mean += w * f.L
        
        H_smooth = expm(L_mean).real.astype(np.float32)
        H_smooth /= (H_smooth[2,2] + 1e-12)
        
        return H_smooth

    def _smooth_partial_window(self) -> np.ndarray:
        if not self._cache:
            return self._prev_H
        
        self._interpolate_logs()
        
        center = len(self._cache) // 2
        weights = np.array([
            f.inliers / max(f.msre, 1e-6) if f.valid else 1e-3
            for f in self._cache
        ])
        
        if weights.sum() < 1e-9:
            weights[:] = 1.0
        
        weights /= weights.sum()
        
        L_mean = np.zeros((3,3), dtype=np.float32)
        for w, f in zip(weights, self._cache):
            if f.L is not None:
                L_mean += w * f.L
        
        H_smooth = expm(L_mean).real.astype(np.float32)
        H_smooth /= (H_smooth[2,2] + 1e-12)
        
        return H_smooth

    def _detect_cut(self, frame: FrameData) -> bool:
        if not self._cache or not frame.valid:
            return False
        
        last_valid = None
        for f in reversed(self._cache):
            if f.valid and f.L is not None:
                last_valid = f
                break
        
        if last_valid is None or frame.L is None:
            return False
        
        diff = np.linalg.norm(frame.L - last_valid.L, 'fro')
        return diff > self.cut_threshold


def _msre(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
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