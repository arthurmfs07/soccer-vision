import cv2
import torch
import numpy as np
from typing import List, Dict
from src.logger import setup_logger

class VideoVisualizer:
    def __init__(self, frame: np.ndarray):
        self.logger = setup_logger("api.log")
        self.frame = self._prepare_frame(frame)


    def annotate(self, annotations: List[Dict]) -> None:
        """Adds bounding boxes and labels to the frame."""
        
        if isinstance(self.frame, torch.Tensor):
            self.frame = self.frame.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        
        if self.frame.max() > 1.0:  # If normalized (0-1), scale back
            self.frame = np.clip(self.frame, 0, 255).astype(np.uint8)
        else:
            self.frame = (self.frame * 255).astype(np.uint8)

        for detection in annotations:
            boxes = detection["boxes"]
            confidences = detection["confidences"]
            classes = detection["classes"]

            for (box, conf, cls) in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to int
                label = f"{int(cls)}: {conf:.2f}"  # Class and confidence
                
                color = (0, 255, 0)  # Green box for detections
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Ensures the frame is in OpenCV compatible format."""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()  # Convert PyTorch tensor to NumPy array
            frame = (frame * 255).clip(0, 255).astype(np.uint8)  # Normalize to [0,255]
        
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)  # Ensure data type is uint8
        
        if frame.shape[-1] == 3:  # Ensure correct color format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        return frame


    def show(self) -> None:
        """Displays the video frame visualization."""
        cv2.imshow("Video Visualizer", self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_image(self) -> np.ndarray:
        """Returns the numpy array of the video frame with annotations."""
        return self.frame