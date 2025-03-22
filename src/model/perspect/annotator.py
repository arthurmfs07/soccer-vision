import cv2
import numpy as np
from typing import List, Literal, Tuple
from src.visual.field import FieldVisualizer, PitchConfig
from src.process.process import Process
from src.visual.visualizer import Visualizer


class HomographyAnnotator(Process):
    """Handles manual annotation of homography points without visualization logic."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.field = FieldVisualizer()
        self.field.frame.resize_to_width(self.image.shape[1])

        self.reference_field_pts = self.field.get_template_pixel_points()

        self.captured_video_pts: List[Tuple[int, int]] = []
        self.sampled_video_pts: List[Tuple[int, int]] = []
        self.video_detection_pts: List[Tuple[int, int]] = []

        self.projected_field_pts: List[Tuple[int, int]] = []
        self.projected_detection_pts: List[Tuple[int, int]] = []

        self.n_points: int = 4
        self.reference_field_indices: List[int] = []
        self.H: np.ndarray = None

        self.phase: Literal["collect_video", "collect_template", "done"] = "collect_video"
        self.active_input_index: int = 0

    def on_mouse_click(self, x: int, y: int) -> None:
        """Captures image points on click during the annotation phase."""
        if self.phase == "collect_video":
            if len(self.captured_video_pts) < self.n_points:
                self.captured_video_pts.append((x, y))
                print(f"Captured image point {len(self.captured_video_pts)}: ({x}, {y})")

            if len(self.captured_video_pts) == 4:
                print("\nAll 4 image points captured.")
                self.phase = "collect_indices"

    def prompt_for_template_indices(self) -> bool:
        """Prompts the user to input template point indices for homography."""
        if self.phase != "collect_indices":
            print("Not in the correct phase to collect template indices.")
            return False

        indices = []
        for i in range(self.n_points):
            idx = int(input(f"Enter template index for image point {i+1}: "))
            assert type(idx) == int and 0 <= idx < len(self.reference_field_pts)
            indices.append(int(idx))
        
        self.reference_field_indices = indices
        print("Template indices received. Now computing homography...")        
        return self.compute_homography()


    def compute_homography(self) -> bool:
        if len(self.captured_video_pts) != 4 or len(self.reference_field_indices) != 4:
            print("Cannot compute homography. 4 points and 4 indices are required.")
            return False

        captured_pts = np.array(self.captured_video_pts, dtype=np.float32)
        reference_pts = self.reference_field_pts[self.reference_field_indices, :]

        self.H_video2field, status = cv2.findHomography(captured_pts, reference_pts, cv2.RANSAC)
        self.H_field2video = np.linalg.inv(self.H_video2field)

        if self.H_video2field is None:
            print("❌ Homography computation failed.")
            return False

        print("Homography computed successfully:")
        print(self.H_video2field)

        self.phase = "done"

        return True

    def save_homography(self, path: str = "homography_video2field_matrix.npy") -> None:
        if self.H_video2field is not None:
            np.save(path, self.H_video2field)
            print(f"💾 Saved homography matrix to {path}")
        else:
            print("⚠️ No homography to save.")

    def load_homography(self, path: str = "homography_video2field_matrix.npy") -> bool:
        try:
            self.H_video2field = np.load(path)
            print(f"✅ Loaded homography matrix from {path}")
            self.phase = "done"
            return True
        except Exception as e:
            print(f"Failed to load homography: {e}")
            return False

    def is_done(self) -> bool:
        return self.phase == "done"

    def run(self):
        """Runs the interactive visualizer with homography annotation logic."""
        visualizer = Visualizer(PitchConfig(), self.image, process=self)

        def click_callback(event: int, x: int, y: int, flags: int, param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.on_mouse_click(x, y)

        cv2.namedWindow("Visualizer")
        cv2.setMouseCallback("Visualizer", click_callback)

        while not self.is_done():
            visualizer.video_visualizer.clear_annotations()
            visualizer.field_visualizer.clear_annotations()
            visualizer._annotate_frames(
                visualizer.video_visualizer.frame,
                visualizer.field_visualizer.frame
            )

            combined_img = visualizer.generate_combined_view()
            cv2.imshow("Visualizer", combined_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.is_done():
                break
            elif len(self.captured_video_pts) >= self.n_points:
                self.prompt_for_template_indices()

        self._generate_projection(visualizer)


    def _generate_projection(self, visualizer):
        """
        Sample 6 points from the center of the video frame,
        projects using homography, and draw both set of points."""
        
        if self.H_video2field is None:
            print("Cannot visualize projection: Homography is None")
            return
        
        img_h, img_w = self.image.shape[:2]
        mean = [img_w / 2, img_h / 2]
        std_dev = [img_w / 8, img_h / 8]


        sampled_video_pts = np.random.normal(loc=mean, scale=std_dev, size=(6,2)).astype(np.float32)

        field_projected_pts = cv2.perspectiveTransform(
            sampled_video_pts.reshape(-1, 1, 2),
            self.H_video2field,
        ).reshape(-1, 2)

        self.sampled_video_pts = sampled_video_pts
        self.projected_field_pts = field_projected_pts

        while True:
            visualizer.video_visualizer.clear_annotations()
            visualizer.field_visualizer.clear_annotations()

            visualizer._annotate_frames(
                visualizer.video_visualizer.frame,
                visualizer.field_visualizer.frame
            )

            # video-space sampled points
            combined_img = visualizer.generate_combined_view()
            cv2.imshow("Visualizer", combined_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_path = "reports/screenshot_2025-03-18.png"
    annotator = HomographyAnnotator(frame_path)
    annotator.run()

    if annotator.H is not None:
        annotator.save_homography("homography_matrix.npy")
