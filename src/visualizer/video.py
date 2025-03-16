import cv2
import torch
import numpy as np
from typing import List, Dict, Any
from src.logger import setup_logger

from dataclasses import dataclass

@dataclass
class VideoFrame:
    """Encapsulate a single video frame with annotations."""
    frame_id: int
    timestamp: float
    image: np.ndarray
    detections: List[Dict[str, Any]]


class VideoVisualizer:
    def __init__(
            self, 
            initial_frame: np.ndarray, 
            class_names: Dict[int, str] = None):
        
        self.class_names = class_names if class_names is not None else {}
        self.logger = setup_logger("api.log")
        self.frame = self._prepare_frame(initial_frame)

        # color for each class
        self.color_dict = {
            0: (255, 255, 255),  # ball : white
            1: (  0,   0, 255),  # goalkeeper : blue
            2: (  0, 255,   0),  # player : green
            3: (255, 165,   0),  # referee : orange
        }



    def update(self, video_frame: VideoFrame) -> None:
        """Updates the visualizer with a new frame and applies annotations."""
        self.frame = self._prepare_frame(video_frame.image)
        self._annotate(video_frame.detections)

    def _annotate(self, detections: List[Dict[str, Any]]) -> None:
        """Adds bounding boxes and labels to the frame."""
        if not detections:
            return  # No detections to annotate

        for box, conf, cls in zip(detections.boxes, detections.confidences, detections.classes):
            x1, y1, x2, y2 = map(int, box)
            label_class = self.class_names.get(int(cls), str(int(cls)))
            label = f"{label_class}: {conf:.2f}"
            color = self.color_dict.get(int(cls), (0, 255, 0))
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(self.frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Converts the frame into OpenCV format."""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)

        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def show(self) -> None:
        """Displays the annotated frame."""
        cv2.imshow("Video Visualizer", self.frame)
        cv2.waitKey(1)

    def get_image(self) -> np.ndarray:
        """Returns the annotated frame as a NumPy array."""
        return self.frame