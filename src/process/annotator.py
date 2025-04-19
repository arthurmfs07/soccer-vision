import cv2
import numpy as np
from time import sleep
import os
from pathlib import Path

from typing import List, Literal, Tuple, Optional
from src.utils import get_data_path
from src.visual.field import FieldVisualizer, PitchConfig
from src.process.process import Process
from src.visual.visualizer import Visualizer
from src.struct.shared_data import SharedAnnotations
from src.logger import setup_logger


class HomographyAnnotator(Process):

    """
    Handles manual annotation of homography points.
    Intended to work over a single image.
    """

    def __init__(
            self, 
            image_path: Optional[str]        = None, 
            image_np:   Optional[np.ndarray] = None,
            visualizer: Optional[Visualizer] = None,
            shared_data: SharedAnnotations   = None,
            output_dir: Optional[str]        = None
            ):

        """Either image_path or image_np, not both."""

        self.visualizer = visualizer
        self.logger = setup_logger("annotator.log")
        self.output_dir = output_dir

        if visualizer is not None:
            self.window_name = self.visualizer.window_name 

        self.image_path = image_path

        if image_path is None and image_np is None:
            raise ValueError(f"No input for image_path or image_np")
        if image_path is not None and image_np is not None:
            raise ValueError("Either image_path or image_np")
        
        if image_np is not None:
            self.image = image_np
        if image_path is not None:
            self.image = cv2.imread(image_path)
        
        self.field = FieldVisualizer()
        self.field.frame.resize_to_width(self.image.shape[1])

        self.shared_data = shared_data
        self.shared_data.reference_field_pts = self.field.get_template_pixel_points()

        self.n_points: int = 4
        self.phase: Literal["collect_video", "collect_template", "done"] = "collect_video"

        self.H_video2field = None
        self.H_field2video = None


    def on_mouse_click(self, x: int, y: int) -> None:
        """Captures image points on click during the annotation phase."""
        if self.phase == "collect_video":
            if len(self.shared_data.captured_video_pts) < self.n_points:
                self.shared_data.captured_video_pts.append((x, y))
                self.logger.info(f"Captured image point {len(self.shared_data.captured_video_pts)}: ({x}, {y})")

            if len(self.shared_data.captured_video_pts) == 4:
                self.logger.info("\nAll 4 image points captured.")
                self.phase = "collect_indices"

    def prompt_for_template_indices(self) -> bool:
        """Prompts the user to input template point indices for homography."""
        indices = []
        for i in range(self.n_points):
            idx = int(input(f"Enter template index for image point {i+1}: "))
            assert type(idx) == int and 0 <= idx < len(self.shared_data.reference_field_pts)
            indices.append(idx)
        
        self.shared_data.reference_field_indices = indices
        self.logger.info("Template indices received. Now computing homography...")        
        return self.compute_homography()


    def compute_homography(self) -> bool:
        if len(self.shared_data.captured_video_pts) != 4 or len(self.shared_data.reference_field_indices) != 4:
            self.logger.info("Cannot compute homography. 4 points and 4 indices are required.")
            return False

        captured_pts = np.array(self.shared_data.captured_video_pts, dtype=np.float32)
        reference_pts = self.shared_data.reference_field_pts[self.shared_data.reference_field_indices, :]

        self.H_video2field, _ = cv2.findHomography(captured_pts, reference_pts, cv2.RANSAC)
        self.H_field2video = np.linalg.inv(self.H_video2field)

        if self.H_video2field is None:
            self.logger.info("❌ Homography computation failed.")
            return False
        
        self.logger.info("Homography computed successfully:")
        self.logger.info(f"\n{self.H_video2field}")
        self.phase = "done"
        return True

    def is_done(self) -> bool:
        return self.phase == "done"

    def run(self, idx: int = 0) -> None:
        """Runs the interactive visualizer with homography annotation logic."""
        if self.visualizer is None:
            self.visualizer = Visualizer(PitchConfig(), self.image, process=self)

        def click_callback(event: int, x: int, y: int, flags: int, param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.on_mouse_click(x, y)

        cv2.namedWindow(self.visualizer.window_name)
        cv2.setMouseCallback(self.visualizer.window_name, click_callback)

        while self.phase in ["collect_video", "collect_indices"]:
            self.visualizer.video_visualizer.clear_annotations()
            self.visualizer.field_visualizer.clear_annotations()
            self.visualizer._annotate_frames(
                self.visualizer.video_visualizer.frame,
                self.visualizer.field_visualizer.frame
            )

            combined_img = self.visualizer.generate_combined_view()
            cv2.imshow(self.visualizer.window_name, combined_img)
            key = cv2.waitKey(1) & 0xFF
            if self.output_dir is None and key == ord('q'):
                break

            if self.phase == "collect_indices":
                self.prompt_for_template_indices()

            self._generate_projection(self.visualizer)

        if self.output_dir is not None:
            answer = input("Do you want to save this annotated frame? (Y/N): ").strip().lower()
            if answer == "y":
                self.save_results(count=idx, output_dir=self.output_dir)
                self.logger.info("Annotated frame saved.")
            else:
                self.logger.info("Annotated frame not saved.")
        else:
            self._generate_projection(self.visualizer)

        cv2.destroyAllWindows()
        self.shared_data.captured_video_pts = []
        self.visualizer.video_visualizer.clear_annotations()
        self.shared_data.H_video2field = self.H_video2field


    def save_results(self, count: int = 0, output_dir: str = "results"):
        output_dir = Path(output_dir)
        if self.H_video2field is not None:
            out_img_path = output_dir / f"annot_frame_{count}.png"
            out_h_path   = output_dir / f"annot_frame_{count}_H.npy"

            cv2.imwrite(str(out_img_path), self.image)
            np.save(str(out_h_path), self.H_video2field)
            self.logger.info(f"✅ Wrote annotated frame + homography to:\n  {out_img_path}\n  {out_h_path}")


    def _generate_projection(self, visualizer):
        """
        Sample 6 points from the center of the video frame,
        projects using homography, and draw both set of points."""
        
        if self.H_video2field is None:
            return
        
        img_h, img_w = self.image.shape[:2]
        mean = [img_w / 2, img_h / 2]
        std_dev = [img_w / 8, img_h / 8]

        sampled_video_pts = np.random.normal(loc=mean, scale=std_dev, size=(6,2)).astype(np.float32)
        field_projected_pts = cv2.perspectiveTransform(
            sampled_video_pts.reshape(-1, 1, 2),
            self.H_video2field,
        ).reshape(-1, 2)

        self.shared_data.sampled_video_pts = sampled_video_pts.tolist()
        self.shared_data.projected_field_pts = [tuple(pt) for pt in field_projected_pts]

        while True:
            visualizer.video_visualizer.clear_annotations()
            visualizer.field_visualizer.clear_annotations()

            visualizer._annotate_frames(
                visualizer.video_visualizer.frame,
                visualizer.field_visualizer.frame
            )

            combined_img = visualizer.generate_combined_view()
            cv2.imshow(self.visualizer.window_name, combined_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            sleep(0.03) # 30ms delay to reduce CPU and avoid GTK freeze

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import random
    frames_dir = Path("data/frames")
    output_dir = "data/annotated_homographies"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not any(frames_dir.iterdir()):
        import subprocess
        print(f"Frames folder is empty. Running frames extraction script...")
        subprocess.run(["python3", "-m", "src.model.perspect.get_frames"], check=True)
    else:
        print("Frames folder is not empty. Proceeding with annotation.")

    match_folders = sorted([f for f in frames_dir.iterdir() if f.is_dir()])
    
    MATCH = 4
    frame_files = list(match_folders[MATCH].glob("*.jpg"))

    n = 20
    random.seed(32 * n)
    
    sampled_frames = random.sample(frame_files, min(n, len(frame_files)))

    for idx, frame_path in enumerate(sampled_frames):
        print(f"\n Processing frame: {frame_path}")

        annotator = HomographyAnnotator(
            image_path=str(frame_path),
            shared_data=SharedAnnotations(),
            output_dir=output_dir
            )
        
        annotator.run(idx)

        print(f"Frame {frame_path.name} processed.")

    print("All frames processed.")


