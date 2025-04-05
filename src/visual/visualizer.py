import cv2
from time import time
import numpy as np
from typing import Optional, Dict
from src.logger import setup_logger
from src.visual.field import FieldVisualizer, PitchConfig
from src.visual.video import VideoVisualizer
from src.struct.frame import Frame
from src.struct.shared_data import SharedAnnotations

class Visualizer:
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

        self.window_name = "Unified Visualization"

        self._resize_frames()
        self._create_H_matrices()


    def get_image(self) -> np.ndarray:
        """Returns the numpy array of the combined visualization."""
        return self.generate_combined_view()

    def update(self, video_frame: "VideoFrame"):
        """Updates the vieo visualization with YOLO detections."""
        self.video_visualizer.update(video_frame)

    def generate_combined_view(self) -> np.ndarray:
        """Generates a combined visualization with the video on top and field below."""

        self.video_visualizer.frame.update_currents()
        self.field_visualizer.frame.update_currents()

        self._resize_frames()

        video_img = self.video_visualizer.frame.rendered
        field_img = self.field_visualizer.frame.rendered

        if video_img.shape[1] != field_img.shape[1]:
            target_width = video_img.shape[1]
            aspect_ratio = field_img.shape[0] / field_img.shape[1]
            new_height = int(round(target_width * aspect_ratio))
            field_img = cv2.resize(field_img, (target_width, new_height), interpolation=cv2.INTER_LINEAR)

        combined_image = np.vstack((video_img, field_img))
        self.frame = Frame(combined_image)
        return combined_image


    def show(self) -> None:
        """Runs the interactive visualization loop, handling user input."""
        def click_callback(event: int, x: int, y: int, flags: int, param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and self.process:
                
                self.process.on_mouse_click(x, y)

        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, click_callback)

        while True:
            if self.process:

                self.video_visualizer.clear_annotations()
                self.field_visualizer.clear_annotations()

                self._annotate_frames(
                    self.video_visualizer.frame, 
                    self.field_visualizer.frame
                )

                combined_img = self.generate_combined_view()
                cv2.imshow(self.window_name, combined_img.data.image)
            
                key = cv2.waitKey(1)
                if key == ord('q') or (self.process and self.process.is_done()):
                    break


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

            self._clear_shared_data()

        combined_img = self.generate_combined_view()
        cv2.imshow(self.window_name, combined_img)
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

        shared = self.process.shared_data if hasattr(self.process, "shared_data") else None
        if shared is None:
            return

        # video_detection_pts = NOT YET IMPLEMENTED
        # for i, (x, y) in enumerate(shared.video_detection_pts):
        #     video_frame.add_circle(x=x, y=y, color="red", radius=6, thickness=-1)
        #     video_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.6)

        # captured video points
        for i, (x, y) in enumerate(shared.captured_video_pts):
            video_frame.add_circle(x=x, y=y, color="red", radius=6, thickness=-1)
            video_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.6)

        # sampled videos points
        for i, (x, y) in enumerate(shared.sampled_video_pts):
            video_frame.add_circle(x=x, y=y, color="blue", radius=6, thickness=-1)
            video_frame.add_text(x + 5, y - 5, f"{i+1}", color="blue", size=0.6)

        # projected_detected_model_pts
        for i, (x, y) in enumerate(shared.projected_detection_model_pts):
            field_frame.add_circle(x=x, y=y, color="blue", radius=2, thickness=-1, coord_space="model")
            # field_frame.add_text(x + 5, y - 5, f"{i+1}", color="blue", size=0.1, coord_space="model")

        # ground truth points
        for i, (x, y) in enumerate(shared.ground_truth_pts):
            field_frame.add_circle(x=x, y=y, color="red", radius=1, thickness=-1, coord_space="model")
            # field_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.4, coord_space="model")

        # projected field points
        for i, (x, y) in enumerate(shared.projected_field_pts):
            field_frame.add_circle(x=x, y=y, color="red", radius=4, thickness=-1, coord_space="current")
            # field_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.4, coord_space="current")

        # projected_detected_pts
        for i, (x, y) in enumerate(shared.projected_detection_pts):
            field_frame.add_circle(x=x, y=y, color="red", radius=4, thickness=-1, coord_space="current")
            field_frame.add_text(x + 5, y - 5, f"{i+1}", color="red", size=0.4, coord_space="current")

        if self.process.is_done():
            video_frame.add_text(10, 20, "Homography ready", color="yellow", size=0.6)


    def _clear_shared_data(self):
        self.process.shared_data.captured_video_pts.clear()
        self.process.shared_data.sampled_video_pts.clear()
        self.process.shared_data.projected_field_pts.clear()
        self.process.shared_data.projected_detection_pts.clear()


    def _resize_frames(self):
        max_width = max(
            self.video_visualizer.frame.current_width, 
            self.field_visualizer.frame.current_width
            )
        
        self.video_visualizer.frame.resize_to_width(max_width)
        self.field_visualizer.frame.resize_to_width(max_width)

    def _create_H_matrices(self):
        self.video_visualizer.frame.update_currents()
        video_height = self.video_visualizer.frame.current_height

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
