import numpy as np
import cv2
from typing import Tuple
from src.visual.field import PitchConfig


class TransformUtils:
    """Utility class for pixel<->metre and homography transformations."""

    @staticmethod
    def px_to_metre(
        pts_px: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert Nx2 array of pixel coords (x_px,y_px) in template
        to meters on the pitch
        """
        h_px, w_px = frame_shape
        sx = PitchConfig().length / w_px
        sy = PitchConfig().width  / h_px
        return pts_px * np.array([sx, sy], dtype=pts_px.dtype)
    

    @staticmethod
    def metre_to_px(
        pts_m: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert Nx2 array of meter coords (x_m,y_m) on the pitch
        to pixel coords in template image.
        """
        h_px, w_px = frame_shape
        sx = w_px / PitchConfig().length
        sy = h_px / PitchConfig().width
        return pts_m * np.array([sx, sy], dtype=pts_m.dtype)


    @staticmethod
    def compute_px_to_px_homography(
        src_pts_px: np.ndarray,
        dst_pts_px: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute a homography mapping video-pixel coords 
        to pitch-metre coords.

        H_m such that for any video-pixel [x,y,1]^T,
            H_m @ [x, y, 1].T -> [X_m, Y_m, _].T (in metres).
        """
        # convert destination template pixels to field metres
        dst_m = TransformUtils.px_to_metre(dst_pts_px, frame_shape)
        H_m, _ = cv2.findHomography(
            src_pts_px.astype(np.float32),
            dst_m.astype(np.float32),
            cv2.RANSAC
        )
        return H_m

    @staticmethod
    def metre_to_px_homography(
        H_m: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Given H_m that maps video pixels -> meters, convert it into
        H_px that maps video pixels -> template pixels.
        """
        h_px, w_px = frame_shape
        S = np.diag([
            w_px / PitchConfig().length,
            h_px / PitchConfig().width,
            1.0
        ])
        return S @ H_m


    @staticmethod
    def get_base_square_px(frame_shape: Tuple[int,int]) -> np.ndarray:
        """
        Returns a 4×2 array of pixel coords for the canonical
        “square” (centre third of the frame).
        """
        h_px, w_px = frame_shape
        s    = h_px / 3.0
        cx   = w_px / 2.0
        cy   = 2 * h_px / 3.0
        return np.array([
            [cx - s/2, cy - s/2],
            [cx + s/2, cy - s/2],
            [cx + s/2, cy + s/2],
            [cx - s/2, cy + s/2],
        ], dtype=np.float32)