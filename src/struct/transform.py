import numpy as np
import cv2
from typing import Tuple


class TransformUtils:
    """Utility class for pixel<->metre and homography transformations."""
   

    @staticmethod
    def px_to_norm(
        pts_px: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert pixle coords (x_px,y_px) -> normalized [0,1] coords.
        """
        h_px, w_px = frame_shape
        return pts_px / np.array([w_px, h_px], dtype=pts_px.dtype)
    
    @staticmethod
    def norm_to_px(
        pts_norm: np.ndarray,
        disp_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        disp_size: (W_px, H_px)
        Convert normalized [0,1] coords -> pixel coords (x_px,y_px).
        """
        w_disp, h_disp = disp_size
        return pts_norm * np.array([w_disp, h_disp], dtype=pts_norm.dtype)