import torch
import numpy as np
from typing import Tuple, List, Union

from src.config import BaseConfig
from src.struct.detection import Detection


def create_base_square(
        as_tensor: bool = True
        ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the base square from video pixel coordinates.
    image_shape: [B, C, H, W]
    Returns a tensor of shape [B, 4, 2]
    or numpy array of shape [4, 2]
    """
    s = 1./4.5
    center_x = 1 / 2
    center_y = 3 / 5
    base_square = [
        [center_x - s / 2.0, center_y - s / 2.0],
        [center_x + s / 2.0, center_y - s / 2.0],
        [center_x + s / 2.0, center_y + s / 2.0],
        [center_x - s / 2.0, center_y + s / 2.0]
    ]
    if as_tensor:
        base_tensor = torch.tensor(
            base_square, dtype=torch.float32, device=BaseConfig().device
            ).unsqueeze(0).expand(1, -1, -1)
        return base_tensor # [B,4,2]
    
    return np.array(base_square, dtype=np.float32) # [4,2]




def get_color(color_name: str) -> Tuple[int, int, int]:
    """Convert color name to BGR tuple."""
    colors = {
        "red":        (0, 0, 255),
        "green":      (0, 128, 0),
        "lightgreen": (144, 238, 144),
        "blue":       (255, 0, 0),     
        "lightblue":  (173, 216, 230),
        "yellow":     (0, 255, 255),
        "orange":     (0, 165, 255),   
        "purple":     (128, 0, 128),   
        "black":      (0, 0, 0),       
        "white":      (255, 255, 255), 
        "offwhite":   (210, 210, 210), 
        "gray":       (128, 128, 128),
        "darkgray":   (50, 50, 50),    
    }
    return colors.get(color_name.lower(), (0, 0, 0))  # Default to black



def annotate_frame_with_detections(
        frame: "Frame",
        detections: List[Detection] 
        ) -> None:
    """Convert YOLO detection into annotations"""

    for detection in detections:
        for i in range(len(detection.boxes)):
            x1, y1, x2, y2 = detection.boxes[i]
            class_id = int(detection.classes[i])
            conf = detection.confidences[i]

            label = f"{frame.data.metadata.get(class_id, str(class_id))}: {conf:.2f}"
            color = "green" if class_id != 0 else "orange"

            frame.add_rect(x1, y1, x2, y2, color=color)
            frame.add_text(x1, y1 - 5, label, color=color)

    return frame