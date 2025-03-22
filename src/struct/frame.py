import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from src.model.detect.objdetect import Detection
from src.struct.annotation import *
from src.struct.utils import *


@dataclass
class FrameData:
    """
    A dataclass that holds image data and annotations.
    """
    image: np.ndarray  # (H, W, C), RGB format
    annotations: List[Annotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.image.ndim == 3 and self.image.shape[2] == 3, \
            f"Expected an image of shape (H, W< 3), but got {self.image.shape}"
        self.metadata["original_shape"] = self.image.shape


class Frame:
    """
    Performs operations on FrameData with proper annotation handling.
    """
    def __init__(
            self, 
            np_image:    np.ndarray,
            annotations: Optional[List[Annotation]] = None,
            metadata:    Optional[Dict[str, Any]] = None
            ):
        """
        np_image expected: (H, W, C), uint8, RGB format.
        """
        
        if annotations is None:
            annotations = []
        if metadata is None:
            metadata = {}

        self.data = FrameData(np_image, annotations, metadata)

        self._original_image = self.data.image.copy()
        self._original_annotations = self.data.annotations.copy()
        self._static_count = 0

        self._scale = 1.0
        self._padding = (0, 0, 0, 0)  # top, bottom, left, right

        self._transform_dirty = True
        self._cached_transformed_image = None
        self._cached_transformed_annotations = None


        h, w, c = self._original_image.shape
        self.original_width = self.current_width = w
        self.original_height = self.current_height = h
        self.channels = c

    @property
    def shape(self):
        self._shape = (self.current_height, self.current_width, 3)
        return self._shape


    @property
    def rendered(self) -> np.ndarray:
        """
        Returns the rendered image with all transformations and annotations applied.
        """
        self._update_transforms_if_needed()
        
        result = self._cached_transformed_image.copy()
        for annotation in self._cached_transformed_annotations:
            annotation.draw(result)
        
        return result

    @property
    def numpy(self) -> np.ndarray:
        """
        Returns the transformed image as a raw numpy array, RGB (H, W, C).
        """
        self._update_transforms_if_needed()
        return self._cached_transformed_image.copy()

    @property
    def tensor(self) -> torch.Tensor:
        """
        Returns the transformed image as a normalized torch tensor (C, H, W), float32.
        """
        self._update_transforms_if_needed()
        tensor = torch.from_numpy(self._cached_transformed_image).permute(2, 0, 1).float() / 255.0
        return tensor


    def update_scale(self, scale: float):
        """Update scaling factor for the image."""
        self._scale = scale
        self._transform_dirty = True
        return self

    def update_padding(self, padding: Tuple[int, int, int, int]):
        """Update padding (top, bottom, left, right)."""
        self._padding = padding
        self._transform_dirty = True
        return self
    
    def clear_annotations(self) -> None:
        self._original_annotations = self._original_annotations[:self._static_count]
        self._transform_dirty = True
        
    def annotation_from_detection(self, detection: Detection) -> None:
        """Convert YOLO detection into annotations"""
        for i in range(len(detection.boxes)):
            x1, y1, x2, y2 = detection.boxes[i]
            class_id = int(detection.classes[i])
            conf = detection.confidences[i]

            label = f"{self.data.metadata.get(class_id, str(class_id))}: {conf:.2f}"
            color = "green" if class_id != 0 else "orange"

            self.add_rect(x1, y1, x2, y2, color=color)
            self.add_text(x1, y1 - 5, label, color=color)


    def resize_to_width(self, new_width: int):
        self.update_currents()
        current_width = self.current_width
        if current_width <= 0:
            return
        
        scale_factor = float(new_width) / float(current_width)
        new_scale = self._scale * scale_factor
        self.current_width = int((scale_factor / float(current_width)))
        self.update_scale(new_scale)


    def set_static_checkpoint(self) -> None:
        """Record how many annotations we currently have."""
        self._static_count = len(self._original_annotations)


    def add_box(self, x: float, y: float, width: float, height: float, label: Optional[str] = None, 
                color: str = "green", coord_space: Literal["model", "current"] = "model"):
        """Add a box annotation."""
        annotation = BoxAnnotation(x, y, width, height, label, color)
        self._add_annotation(annotation, coord_space)

    def add_point(self, x: float, y: float, color: str = "red", radius: int = 5,
                 coord_space: Literal["model", "current"] = "model"):
        """Add a point annotation."""
        annotation = PointAnnotation(x, y, color, radius)
        self._add_annotation(annotation, coord_space)

    def add_circle(self, x: float, y: float, radius: int = 5, color: str = "red", 
                  thickness: int = 1, coord_space: Literal["model", "current"] = "model"):
        """Add a circle annotation."""
        annotation = CircleAnnotation(x, y, radius, color, thickness)
        self._add_annotation(annotation, coord_space)

    def add_text(self, x: float, y: float, text: str, color: str = "black", 
                size: float = 0.5, coord_space: Literal["model", "current"] = "model"):
        """Add a text annotation."""
        annotation = TextAnnotation(x, y, text, color, size)
        self._add_annotation(annotation, coord_space)
    
    def add_line(self, x1: float, y1: float, x2: float, y2: float, color: str = "green", 
                thickness: int = 1, coord_space: Literal["model", "current"] = "model"):
        """Add a line annotation."""
        annotation = LineAnnotation(x1, y1, x2, y2, color, thickness)
        self._add_annotation(annotation, coord_space)

    def add_rect(self, x1: float, y1: float, x2: float, y2: float, color: str = "green",
                thickness: int = 1, coord_space: Literal["model", "current"] = "model"):
        """Add a rectangle annotation."""
        annotation = RectTLAnnotation(x1, y1, x2, y2, color, thickness)
        self._add_annotation(annotation, coord_space)
        

    def _add_annotation(self, annotation: object, coord_space: Literal["model", "current"] = "model") -> 'Frame':
        """
        Add any type of annotation to the frame.
        
        Args:
            annotation: The annotation to add
            coord_space: Whether the coordinates are in "model" space (original image) 
                         or "current" space (already transformed)
        """
        if coord_space == "current":
            top_offset = self._padding[0]
            left_offset = self._padding[2]
            annotation = self._inverse_transform_annotation(annotation, self._scale, left_offset, top_offset)            
        self._original_annotations.append(annotation)
        self._transform_dirty = True
        return self
    
    def _inverse_transform_annotation(self, annotation: object, scale: float, offset_x: int, offset_y: int) -> object:
        attrs = annotation.__dict__.copy()
        for attr_name, value in attrs.items():
            if attr_name in ('color', 'thickness', 'text', 'label', 'size', 'coord_space'):
                continue
                
            if attr_name in ('x', 'x1', 'x2'):
                attrs[attr_name] = (value - offset_x) / scale
            elif attr_name in ('y', 'y1', 'y2'):
                attrs[attr_name] = (value - offset_y) / scale
            elif attr_name in ('width', 'height', 'radius'):
                attrs[attr_name] = value / scale
        
        # Create a new instance of the same class with transformed attributes
        return type(annotation)(**attrs)
        

    @property
    def shape(self):
        self._shape = (self.current_height, self.current_width, 3)
        return self._shape
    
    def _update_transforms_if_needed(self):
        """Update transformations if dirty or not yet calculated."""
        if self._transform_dirty or self._cached_transformed_image is None:
            self._cached_transformed_image = self._apply_transforms(self._original_image.copy())
            self._cached_transformed_annotations = self._transform_annotations()
            self._transform_dirty = False

    def update_currents(self):
        if self._transform_dirty or self._cached_transformed_image is None:
            w, h = self.original_width, self.original_height
            new_w = round(w * self._scale)
            new_h = round(h * self._scale)

            self.current_width = new_w
            self.current_height = new_h

        top, bottom, left, right = self._padding
        if any(p > 0 for p in self._padding):
            self.current_width += (left + right)
            self.current_height += (top + bottom)


    def _apply_transforms(self, img: np.ndarray) -> np.ndarray:
        """
        Applies scale and padding in order.
        """
        h, w = self.original_height, self.original_width
        if self._scale != 1.0:
            new_w = round(w * self._scale)
            new_h = round(h * self._scale)
            img = cv2.resize(img, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        else:
            new_w, new_h = w, h

        self.current_width = new_w
        self.current_height = new_h

        top, bottom, left, right = self._padding
        if any(p > 0 for p in self._padding):
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=get_color(self.data.metadata["bgc"]))
            self.current_width += (left + right)
            self.current_height += (top + bottom)

        return img

    def _transform_annotations(self) -> List[Annotation]:
        """
        Transforms all annotations according to the current transformation parameters.
        """
        top_offset = self._padding[0]
        left_offset = self._padding[2]
        transformed = []
        for annotation in self._original_annotations:
            transformed.append(annotation.transform(self._scale, left_offset, top_offset))
        return transformed