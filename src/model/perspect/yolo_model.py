import cv2
import torch
import warnings
from ultralytics import YOLO
from typing import *
import numpy as np

from src.logger import setup_logger
from src.struct.utils              import create_base_square

from src.utils import rf2my
from src.model.perspect.homography import HomographyEstimator


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
        iou: float  = 0.45
    ):

        self.device = device
        self.imgsz  = imgsz
        self._yolo = YOLO(weights)
        self._yolo.to(device)
        self._yolo.overrides = dict(conf=0.9, iou=iou)
        self.K = int(self._yolo.model.yaml["kpt_shape"][0])

        self.logger = setup_logger("yolo_model.log")

        self.conf_thr:       float = 0.45
        self.border_margin: float = 0.01

        self.homography_estimator = HomographyEstimator(device=self.device)

        inv = [0] * self.K
        for old, new in rf2my.items():
            inv[new] = old
        self.inv = torch.as_tensor(inv, dtype=torch.long)   


    def eval(self):
        """No-op to unify with nn.Module.eval() calls."""
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

        coords_list: List[torch.Tensor] = []
        vis_list:    List[torch.Tensor] = []

        for r in results:
            # r.keypoints: Tensor[N, K, 3] or None
            if (
                r.keypoints is None 
                or r.keypoints.shape[0] == 0
                or getattr(r.keypoints, "conf", None) is None
                or r.keypoints.conf.shape[0] == 0
            ):
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


        for b in range(B):
            coords_list[b] = coords_list[b][self.inv]
            vis_list[b]    = vis_list[b][self.inv]

        # stack into batched tensors
        coords   = torch.stack(coords_list)  # (B,K,2)
        vis_prob = torch.stack(vis_list)     # (B,K)

        m = self.border_margin
        x = coords[..., 0]
        y = coords[..., 1]
        border_mask = (x < m) | (x > (1.0 - m)) | (y < m) | (y > (1.0 - m))
        vis_prob = vis_prob.masked_fill(border_mask, 0.0)

        if not predict_H:
            return None, coords, vis_prob
        
        Hs = self.homography_estimator.predict(coords, vis_prob, self.conf_thr)

        return Hs, coords, vis_prob

    __call__ = predict
