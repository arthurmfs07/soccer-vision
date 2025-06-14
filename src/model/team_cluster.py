# src/model/team_cluster.py
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

class TeamClusterer:
    """
    Unsupervised 2-colour clustering of player jerseys using color histograms.
    Always assigns cluster-0 to the darker centroid (Lab-L).
    """

    def __init__(self, crop_top: float = 0.40, n_bins: int = 6) -> None:
        self.crop_top = float(np.clip(crop_top, 0.05, 0.95))
        self.n_bins = n_bins

    def _feat(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Extract histogram features from player bbox."""
        h, w = frame.shape[:2]
        if box.max() <= 1.05:
            box = box * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        y_cut = int(y1 + self.crop_top * (y2 - y1))
        crop = frame[y1:y_cut, x1:x2]
        
        if crop.size == 0:
            return np.zeros(self.n_bins * 3 + 2, np.float32)
        
        # Convert to Lab color space
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        
        # Create weight mask - center pixels more important than edges
        h_crop, w_crop = lab.shape[:2]
        y_weights = np.exp(-((np.arange(h_crop) - h_crop/2) / (h_crop/3))**2)
        x_weights = np.exp(-((np.arange(w_crop) - w_crop/2) / (w_crop/3))**2)
        weights = np.outer(y_weights, x_weights).flatten()
        
        # Reshape for histogram computation
        lab_flat = lab.reshape(-1, 3)
        
        # Compute weighted histogram for each channel
        features = []
        for i in range(3):
            hist, _ = np.histogram(lab_flat[:, i], bins=self.n_bins, range=(0, 256), weights=weights)
            hist = hist.astype(np.float32) / (hist.sum() + 1e-6)  # Normalize
            features.extend(hist)
        
        # Add color ratio features (a/L and b/L) for lighting invariance
        # Use median for robustness
        l_median = np.median(lab_flat[:, 0])
        a_median = np.median(lab_flat[:, 1]) 
        b_median = np.median(lab_flat[:, 2])
        
        if l_median > 20:  # Only if sufficient brightness
            a_ratio = (a_median - 128) / l_median  # Center around 0
            b_ratio = (b_median - 128) / l_median
        else:
            a_ratio = b_ratio = 0
            
        features.extend([a_ratio, b_ratio])
        
        return np.array(features, np.float32)

    def cluster_batch(
        self,
        frames_bgr: List[np.ndarray],
        boxes_list: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns per-frame boolean masks (team-0 = darker, team-1 = lighter)."""
        feats = []
        for frm, boxes in zip(frames_bgr, boxes_list):
            for b in boxes:
                feats.append(self._feat(frm, b))

        X = np.vstack(feats) if feats else np.empty((0, self.n_bins * 3 + 2), np.float32)

        # trivial: <2 players → everyone team-0
        if X.shape[0] < 2:
            return [
                (np.ones(len(b), bool), np.zeros(len(b), bool))
                for b in boxes_list
            ]

        # Use multiple attempts for more stable clustering
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-4)
        _, raw_lbl, cents = cv2.kmeans(
            X, 2, None, term, 5, cv2.KMEANS_PP_CENTERS  # 5 attempts instead of 1
        )
        lbl = raw_lbl.ravel().astype(int)
        c0, c1 = cents
        
        # Determine which cluster is darker based on L channel histogram
        # The L channel is the first n_bins values in each center
        # Calculate weighted average brightness from L histogram
        bin_centers = np.linspace(0, 255, self.n_bins)
        brightness_0 = np.dot(c0[:self.n_bins], bin_centers)
        brightness_1 = np.dot(c1[:self.n_bins], bin_centers)
        
        # swap if c0 is lighter than c1
        if brightness_0 > brightness_1:
            lbl = 1 - lbl
            c0, c1 = c1, c0

        # split back to per-frame masks
        masks, ptr = [], 0
        for boxes in boxes_list:
            n = len(boxes)
            m1 = lbl[ptr:ptr+n].astype(bool)  # lighter = cluster 1
            masks.append((~m1, m1))            # (darker, lighter)
            ptr += n

        return masks