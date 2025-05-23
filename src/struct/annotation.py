# file: src/struct/annotation.py

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.struct.transform import TransformUtils
from src.struct.utils import get_color


class Annotation(ABC):
    """
    Base class for annotations in normalized [0,1]^2 coords.
    Must implement draw(image, disp_size) to denormalize into pixel space.
    """

    @abstractmethod
    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        """
        Draw this annotation on the given image.
        :param image:    HxW BGR image to draw into
        :param disp_size: (W_px, H_px) of that image for denormalization
        """
        ...


@dataclass
class BoxAnnotation(Annotation):
    """
    Normalized box: center (x,y) in [0,1], width/height in [0,1] fraction of canvas.
    """
    x: float
    y: float
    width: float
    height: float
    label: Optional[str] = None
    color: str = "green"
    thickness: float = 0.005  # fraction of width

    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        # center denormalization
        cx, cy = TransformUtils.norm_to_px(
            np.array([[self.x, self.y]]), disp_size
        ).flatten()
        # size denormalization (disp_size = (W, H))
        w_px = int(self.width  * disp_size[0])
        h_px = int(self.height * disp_size[1])
        # compute corners
        pt1 = (int(cx - w_px/2), int(cy - h_px/2))
        pt2 = (int(cx + w_px/2), int(cy + h_px/2))
        t_px = -1 if self.thickness <= 0 else max(
            1, int(self.thickness * disp_size[0])
        )
        cv2.rectangle(image, pt1, pt2, get_color(self.color), t_px)

        if self.label:
            text = self.label
            font = cv2.FONT_HERSHEY_SIMPLEX
            # estimate text size
            ((tw, th), bs) = cv2.getTextSize(text, font, 0.6, 1)
            tx = min(pt1[0] + 5, disp_size[0] - tw - 5)
            ty = max(pt1[1] + th + 5, th + 5)
            cv2.putText(
                image, text,
                (int(tx), int(ty)),
                font, 0.5,
                get_color(self.color), 1
            )


@dataclass
class PointAnnotation(Annotation):
    """
    Normalized point in [0,1]^2.
    """
    x: float
    y: float
    color: str = "red"
    radius: float = 0.01  # fraction of width

    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        cx, cy = TransformUtils.norm_to_px(
            np.array([[self.x, self.y]]), disp_size
        ).flatten()
        r_px = int(self.radius * disp_size[0])
        cv2.circle(image, (int(cx), int(cy)), r_px, get_color(self.color), -1)


@dataclass
class TextAnnotation(Annotation):
    """
    Normalized text with position and content.
    """
    x: float
    y: float
    text: str
    color: str = "black"
    font_scale: float = 0.8
    thickness: float = 1
    margin: float = 0.01

    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        W_px, H_px = disp_size
        (cx, cy) = TransformUtils.norm_to_px(
            np.array([[self.x, self.y]]), disp_size
        ).flatten()

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(
            self.text, font, self.font_scale, self.thickness
            )
        mx = int(self.margin * W_px)
        my = int(self.margin * H_px)
        # horizontal offset: move right if on left half, and vice versa
        if cx < W_px * 0.5:
            tx = int(cx + mx)
        else:
            tx = int(cx - tw - mx)
        # vertical offset: move bottom if on top half, and vice versa
        if cy < H_px * 0.5:
            ty = int(cy + th + my)
        else:
            ty = int(cy - my)

        cv2.putText(
            image, self.text,
            (tx, ty),
            font,
            self.font_scale,
            get_color(self.color),
            self.thickness,
            cv2.LINE_AA
        )


@dataclass
class LineAnnotation(Annotation):
    """
    Normalized line from (x1,y1) to (x2,y2) in [0,1]^2.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "green"
    thickness: float = 0.005  # fraction of width

    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        p1 = TransformUtils.norm_to_px(
            np.array([[self.x1, self.y1]]), disp_size
        ).flatten()
        p2 = TransformUtils.norm_to_px(
            np.array([[self.x2, self.y2]]), disp_size
        ).flatten()
        t_px = -1 if self.thickness <= 0 else max(
            1, int(self.thickness * disp_size[0])
        )
        cv2.line(
            image,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            get_color(self.color),
            t_px
        )


@dataclass
class RectTLAnnotation(Annotation):
    """
    Rectangle specified by top-left (x1, y1) and bottom-right (x2, y2),
    all normalized in [0,1]^2.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "green"
    thickness: float = 0.005  # fraction of width

    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        p1 = TransformUtils.norm_to_px(
            np.array([[self.x1, self.y1]]), disp_size
        ).flatten()
        p2 = TransformUtils.norm_to_px(
            np.array([[self.x2, self.y2]]), disp_size
        ).flatten()
        t_px = -1 if self.thickness <= 0 else max(
            1, int(self.thickness * disp_size[0])
        )   
        cv2.rectangle(
            image,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            get_color(self.color),
            t_px
        )


@dataclass
class CircleAnnotation(Annotation):
    """
    Circle with center (x, y) and radius in normalized units [0,1].
    """
    x: float
    y: float
    radius: float
    color: str = "red"
    thickness: float = 0  # 0 or -1 for filled

    def draw(self, image: np.ndarray, disp_size: Tuple[int, int]) -> None:
        cx, cy = TransformUtils.norm_to_px(
            np.array([[self.x, self.y]]), disp_size
        ).flatten()
        r_px = int(self.radius * disp_size[0])
        t_px = -1 if self.thickness <= 0 else max(
            1, int(self.thickness * disp_size[0])
        )
        cv2.circle(
            image,
            (int(cx), int(cy)),
            r_px,
            get_color(self.color),
            t_px
        )
