import cv2
from time import time
import numpy as np
from typing import Optional, Dict
from src.logger import setup_logger
from src.visual.field import FieldVisualizer, PitchConfig
from src.visual.video import VideoVisualizer
from src.struct.frame import Frame
from src.visual.window import Window


class Visualizer(Window):
    """Manages visualization, integraetes video and field, and interacts with a process."""

    def __init__(
            self, 
            field_config: PitchConfig, 
            frame: np.ndarray, 
            class_names: Dict[int, str] = None, 
            process: Optional["Process"] = None
            ):
        self.logger = setup_logger("api.log")
        self.field_visualizer = FieldVisualizer(field_config)
        self.video_visualizer = VideoVisualizer(frame, class_names)
        self.process = process

        self._resize_frames()
        self._create_H_matrices()


    def get_image(self) -> np.ndarray:
        """Returns the numpy array of the combined visualization."""
        return self.generate_combined_view()

    def update(self, video_frame):
        """Updates the vieo visualization with YOLO detections."""
        self.video_visualizer.update(video_frame)

    def generate_combined_view(self) -> np.ndarray:
        """Generates a combined visualization with the video on top and field below."""

        self.video_visualizer.frame.update_currents()
        self.field_visualizer.frame.update_currents()

        self._resize_frames()

        video_img = self.video_visualizer.frame.rendered
        field_img = self.field_visualizer.frame.rendered

        combined_image = np.vstack((video_img, field_img))
        self.frame = Frame(combined_image)
        return combined_image


    def show(self) -> None:
        """Runs the interactive visualization loop, handling user input."""
        def click_callback(event: int, x: int, y: int, flags: int, param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and self.process:
                
                self.process.on_mouse_click(x, y)

        
        cv2.namedWindow("Visualizer")
        cv2.setMouseCallback("Visualizer", click_callback)

        while True:
            if self.process:

                self.video_visualizer.clear_annotations()
                self.field_visualizer.clear_annotations()

                self._annotate_frames(
                    self.video_visualizer.frame, 
                    self.field_visualizer.frame
                )

                combined_img = self.generate_combined_view()
                cv2.imshow("Visualizer", combined_img.data.image)
            
                time.sleep(.5)

                key = cv2.waitKey(1)
                if key == ord('q') or (self.process and self.process.is_done()):
                    break

        cv2.destroyAllWindows()

    def render(self) -> None:
        """
        Renders a single frame with the current video and field visualizer.
        Used for real-time inference, without looping.
        """
        if self.process:
            self.video_visualizer.clear_annotations()
            self.field_visualizer.clear_annotations()
            self._annotate_frames(
                self.video_visualizer.frame, 
                self.field_visualizer.frame
            )

        combined_img = self.generate_combined_view()
        cv2.imshow("visualizer", combined_img)
        cv2.waitKey(1)


    def transform_points(self, points: np.ndarray, H: np.ndarray):
        """
        Apply H (3, 3) matrix in (N, 2) points (x, y)
        """
        pts_reshaped = points.reshape(-1, 1, 2). astype(np.float32)
        transformed = cv2.perspectiveTransform(pts_reshaped, H)
        return transformed.reshape(-1, 2)


    def _annotate_frames(
            self,
            video_frame: Frame,
            field_frame: Frame,
        ) -> None:

        # video_detection_pts = NOT YET IMPLEMENTED
        if hasattr(self.process, "video_detection_pts"):
            for i, (x, y) in enumerate(self.process.video_detection_pts):
                video_frame.add_circle(x=x, y=y, color="red", radius=6, thickness=-1)
                video_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.6)

        # captured video points
        if hasattr(self.process, "captured_video_pts"):
            for i, (x, y) in enumerate(self.process.captured_video_pts):
                video_frame.add_circle(x=x, y=y, color="red", radius=6, thickness=-1)
                video_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.6)


        # sampled videos points
        if hasattr(self.process, "sampled_video_pts"):
            for i, (x, y) in enumerate(self.process.sampled_video_pts):
                video_frame.add_circle(x=x, y=y, color="blue", radius=6, thickness=-1)
                video_frame.add_text(x + 5, y - 5, f"{i+1}", color="blue", size=0.6)


        # projected field points
        if hasattr(self.process, "projected_field_pts"):
            for i, (x, y) in enumerate(self.process.projected_field_pts):
                field_frame.add_circle(x=x, y=y, color="blue", radius=4, thickness=-1, coord_space="current")
                field_frame.add_text(x + 5, y - 5, f"{i+1}", color="blue", size=0.4)

        # projected_detected_pts
        if hasattr(self.process, "projected_detection_pts"):
            for i, (x, y) in enumerate(self.process.projected_detection_pts):
                field_frame.add_circle(x=x, y=y, color="green", radius=4, thickness=-1, coord_space="current")
                field_frame.add_text(x + 5, y - 5, f"{i+1}", color="blue", size=0.4)

        if self.process.is_done():
            video_frame.add_text(10, 20, "Homography ready", color="yellow", size=0.6)


    def _resize_frames(self):
        max_width = max(
            self.video_visualizer.frame.current_width, 
            self.field_visualizer.frame.current_width
            )
        
        self.video_visualizer.resize_to_width(max_width)
        self.field_visualizer.resize_to_width(max_width)


    def _create_H_matrices(self):
        self.video_visualizer.frame.update_currents()
        video_height = self.video_visualizer.frame.current_width
        video_height -= 200

        self.H_video2combined = np.eye(3)
        
        self.H_combined2video = np.eye(3)

        self.H_combined2field = np.array([
            [1, 0, 0],
            [0, 1, -video_height],
            [0, 0, 1]
        ], dtype=np.float32)

        self.H_field2combined = np.array([
            [1, 0, 0],
            [0, 1, video_height],
            [0, 0, 1]
        ], dtype=np.float32)


if __name__ == "__main__":
    config = PitchConfig(linewidth=1)
    dummy_frame = np.full((300, 400, 3), (0, 0, 0), dtype=np.uint8)  # Placeholder for a real video frame
    visualizer = Visualizer(config, dummy_frame)
    visualizer.show()
