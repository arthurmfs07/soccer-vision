import cv2
import numpy as np

from typing import Dict
from src.logger import setup_logger
from src.visualizer.field import FieldVisualizer, PitchConfig
from src.visualizer.video import VideoVisualizer

class Visualizer:
    def __init__(self, field_config: PitchConfig, frame: np.ndarray, class_names: Dict[int, str] = None):
        self.logger = setup_logger("api.log")

        self.field_visualizer = FieldVisualizer(field_config)
        self.video_visualizer = VideoVisualizer(frame, class_names)

    def update(self, video_frame):
        """Updates the vieo visualization with YOLO detections."""
        self.video_visualizer.update(video_frame)

    def generate_combined_view(self) -> np.ndarray:
        """Generates a combined visualization with the video on top and field below."""
        video_img = self.video_visualizer.get_image()
        field_img = self.field_visualizer.get_image()
        
        max_width = max(video_img.shape[1], field_img.shape[1])
        
        video_img_resized = cv2.resize(video_img, (max_width, video_img.shape[0]))
        field_img_resized = cv2.resize(field_img, (max_width, field_img.shape[0]))
        
        return np.vstack((video_img_resized, field_img_resized))


    def show(self) -> None:
        """
        Displays the latest visualization.
        """
        combined_img = self.generate_combined_view()
        cv2.imshow("Visualizer", combined_img)
        key = cv2.waitKey(1) & 0xFF  
        
        if key == ord('q'):
            cv2.destroyAllWindows()


    def get_image(self) -> np.ndarray:
        """Returns the numpy array of the combined visualization."""
        return self.generate_combined_view()


if __name__ == "__main__":
    config = PitchConfig(scale=5, linewidth=1)
    dummy_frame = np.full((300, 400, 3), (0, 0, 0), dtype=np.uint8)  # Placeholder for a real video frame
    visualizer = Visualizer(config, dummy_frame)
    visualizer.show()
