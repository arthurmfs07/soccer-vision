import numpy as np
import cv2
from typing import Optional, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.struct.utils import *


class Annotation(ABC):
    """Base class for all annotations"""

    coord_space: Literal["model", "pixel"] = "model"

    @abstractmethod
    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'Annotation':
        """Apply transformation and return a new annotation"""
        if self.coord_space == "model":
            x_t = self.x * scale + offset_x
            y_t = self.y * scale + offset_y
            r_t = int(self.radius * scale)
        elif self.coord_space == "pixel":
            x_t = self.x
            y_t = self.y
            r_t = self.radius
        raise NotImplementedError()
    @abstractmethod
    def draw(self, image: np.ndarray) -> None:
        """Draw this annotation on the given image"""
        raise NotImplementedError()


@dataclass
class BoxAnnotation(Annotation):
    """Box annotation with center coordinates, width, height and optional label"""
    x: float
    y: float
    width: float
    height: float
    label: Optional[str] = None
    color: str = "green"
    thickness: int = 1
    
    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'BoxAnnotation':
        return BoxAnnotation(
            x=self.x * scale + offset_x,
            y=self.y * scale + offset_y,
            width=self.width * scale,
            height=self.height * scale,
            label=self.label,
            color=self.color,
            thickness=self.thickness
        )
    
    def draw(self, image: np.ndarray) -> None:
        pt1 = (int(self.x - self.width / 2), int(self.y - self.height / 2))
        pt2 = (int(self.x + self.width / 2), int(self.y + self.height / 2))
        cv2.rectangle(image, pt1, pt2, get_color(self.color), self.thickness)
        if self.label:
            cv2.putText(image, self.label, (pt1[0], pt1[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(self.color), 1)


@dataclass
class PointAnnotation(Annotation):
    """Point annotation with x, y coordinates"""
    x: float
    y: float
    color: str = "red"
    radius: int = 5
    
    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'PointAnnotation':
        return PointAnnotation(
            x=self.x * scale + offset_x,
            y=self.y * scale + offset_y,
            color=self.color,
            radius=self.radius
        )
    
    def draw(self, image: np.ndarray) -> None:
        cv2.circle(image, (int(self.x), int(self.y)), self.radius, get_color(self.color), -1)


@dataclass
class TextAnnotation(Annotation):
    """Text annotation with position and content"""
    x: float
    y: float
    text: str
    color: str = "black"
    size: float = 0.5
    
    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'TextAnnotation':
        return TextAnnotation(
            x=self.x * scale + offset_x,
            y=self.y * scale + offset_y,
            text=self.text,
            color=self.color,
            size=self.size
        )
    
    def draw(self, image: np.ndarray) -> None:
        cv2.putText(image, self.text, (int(self.x), int(self.y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.size, get_color(self.color), 1)


@dataclass
class LineAnnotation(Annotation):
    """Line annotation with start and end points"""
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "green"
    thickness: int = 1
    
    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'LineAnnotation':
        return LineAnnotation(
            x1=self.x1 * scale + offset_x,
            y1=self.y1 * scale + offset_y,
            x2=self.x2 * scale + offset_x,
            y2=self.y2 * scale + offset_y,
            color=self.color,
            thickness=self.thickness
        )
    
    def draw(self, image: np.ndarray) -> None:
        pt1 = (int(self.x1), int(self.y1))
        pt2 = (int(self.x2), int(self.y2))
        cv2.line(image, pt1, pt2, get_color(self.color), self.thickness)


# file: src/struct/annotation_ext.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.struct.annotation import Annotation, get_color

@dataclass
class RectTLAnnotation(Annotation):
    """
    Draw a rectangle specified by top-left (x1, y1) and bottom-right (x2, y2).
    Coordinates are in 'model space' (to be scaled/translated by Frame).
    """
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "green"
    thickness: int = 1  # -1 => filled

    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'RectTLAnnotation':
        return RectTLAnnotation(
            x1=self.x1 * scale + offset_x,
            y1=self.y1 * scale + offset_y,
            x2=self.x2 * scale + offset_x,
            y2=self.y2 * scale + offset_y,
            color=self.color,
            thickness=self.thickness
        )

    def draw(self, image: np.ndarray) -> None:
        pt1 = (int(self.x1), int(self.y1))
        pt2 = (int(self.x2), int(self.y2))
        cv2.rectangle(image, pt1, pt2, get_color(self.color), self.thickness)


@dataclass
class CircleAnnotation(Annotation):
    """
    Draw a circle with center (x, y) in model space and radius in model units.
    The radius is also scaled by 'scale'.
    """
    x: float
    y: float
    radius: float
    color: str = "red"
    thickness: int = 1  # -1 => filled

    def transform(self, scale: float, offset_x: int, offset_y: int) -> 'CircleAnnotation':
        return CircleAnnotation(
            x=self.x * scale + offset_x,
            y=self.y * scale + offset_y,
            radius=self.radius * scale,
            color=self.color,
            thickness=self.thickness
        )

    def draw(self, image: np.ndarray) -> None:
        center = (int(self.x), int(self.y))
        cv2.circle(image, center, int(self.radius), get_color(self.color), self.thickness)
