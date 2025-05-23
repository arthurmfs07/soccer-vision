# file: src/struct/frame.py

import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from src.struct.annotation import (
    Annotation,
    BoxAnnotation,
    PointAnnotation,
    TextAnnotation,
    LineAnnotation,
    RectTLAnnotation,
    CircleAnnotation,
)
from src.struct.utils import get_color


@dataclass
class FrameData:
    """
    Holds raw image and a list of normalized‐coordinate annotations.
    """
    image: np.ndarray                # (H, W, C), uint8 BGR or RGB
    annotations: List[Annotation] = field(default_factory=list)
    metadata:    Dict[str, Any]     = field(default_factory=dict)

    def __post_init__(self):
        assert self.image.ndim == 3 and self.image.shape[2] == 3, \
            f"Expected H×W×3 image, got {self.image.shape}"
        # record original shape if needed
        self.metadata.setdefault("original_shape", self.image.shape)


class Frame:
    """
    A Frame holds a raw image plus normalized‐coord annotations.
    Everything is in [0..1] internally; only at render() time do we
    resize to pixels and draw.
    """

    def __init__(
        self,
        np_image: np.ndarray,
        annotations: Optional[List[Annotation]] = None,
        metadata: Optional[Dict[str, Any]]   = None
    ):
        if annotations is None:
            annotations = []
        if metadata is None:
            metadata = {}

        self.data = FrameData(np_image, annotations, metadata)

    def add_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        label: Optional[str] = None,
        color: str = "green",
        thickness: float = 0.005
    ) -> "Frame":
        """x,y,width,height in [0,1]; all fractions of target canvas."""
        ann = BoxAnnotation(x, y, width, height, label, color, thickness)
        self.data.annotations.append(ann)
        return self

    def add_point(
        self,
        x: float,
        y: float,
        color: str = "red",
        radius: float = 0.01
    ) -> "Frame":
        """x,y in [0,1]; radius fraction of width."""
        ann = PointAnnotation(x, y, color, radius)
        self.data.annotations.append(ann)
        return self

    def add_circle(
        self,
        x: float,
        y: float,
        radius: float = 0.01,
        color: str = "red",
        thickness: float = 0
    ) -> "Frame":
        """Normalized circle."""
        ann = CircleAnnotation(x, y, radius, color, thickness)
        self.data.annotations.append(ann)
        return self

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        color: str = "black",
        size: float = 0.02
    ) -> "Frame":
        """x,y in [0,1]; size fraction of height."""
        ann = TextAnnotation(x, y, text, color, size)
        self.data.annotations.append(ann)
        return self

    def add_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "green",
        thickness: float = 0.005
    ) -> "Frame":
        """Endpoints normalized in [0,1]."""
        ann = LineAnnotation(x1, y1, x2, y2, color, thickness)
        self.data.annotations.append(ann)
        return self

    def add_rect(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "green",
        thickness: float = 0.005
    ) -> "Frame":
        """Top‐left/bottom‐right in normalized coords."""
        ann = RectTLAnnotation(x1, y1, x2, y2, color, thickness)
        self.data.annotations.append(ann)
        return self

    def clear_annotations(self) -> "Frame":
        """Remove all annotations."""
        self.data.annotations.clear()
        return self

    def render(self, disp_size: Tuple[int, int]) -> np.ndarray:
        """
        Produce an H×W×3 uint8 image at disp_size=(W_px, H_px):
          1. Resize raw image
          2. Draw each normalized annotation
        """
        W_px, H_px = disp_size
        # resize raw
        canvas = cv2.resize(
            self.data.image,
            (W_px, H_px),
            interpolation=cv2.INTER_LINEAR
        )
        # draw annotations
        for ann in self.data.annotations:
            ann.draw(canvas, disp_size)
        return canvas

    @property
    def numpy(self) -> np.ndarray:
        """
        For compatibility: returns the raw image as-is.
        Use render(disp_size) to get a pixel canvas.
        """
        return self.data.image.copy()

    def as_tensor(
        self,
        disp_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Returns a float32 tensor (C, H, W) in [0,1], after rendering.
        """
        img = self.render(disp_size)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t

    @property
    def shape(self) -> Tuple[int,int,int]:
        """
        Returns the raw image shape (H, W, 3).
        """
        return self.data.image.shape
