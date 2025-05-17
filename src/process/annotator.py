import cv2
import numpy as np
from time import sleep
from pathlib import Path
from typing import Literal, Optional

from src.utils import get_data_path
from src.visual.field import FieldVisualizer, PitchConfig
from src.process.process import Process
from src.visual.visualizer import Visualizer
from src.struct.shared_data import SharedAnnotations
from src.struct.transform import TransformUtils
from src.struct.utils import create_base_square
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
        self.health_ok: bool = False

        if image_path is None and image_np is None:
            raise ValueError(f"No input for image_path or image_np")
        if image_path is not None and image_np is not None:
            raise ValueError("Either image_path or image_np")
        
        if image_np is not None:
            self.image = image_np
        if image_path is not None:
            self.image = cv2.imread(image_path)
        
        self.h_px : int = self.image.shape[0]
        self.w_px : int = self.image.shape[1]

        self.field = FieldVisualizer()
        self.field.frame.resize_to_width(self.w_px)
        self.field.frame.update_currents()

        self.tpl_h_px = self.field.frame.current_height
        self.tpl_w_px = self.field.frame.current_width

        self.shared_data = shared_data
        self.shared_data.reference_field_pts = self.field.get_template_pixel_points()

        self.reference_field_meter = TransformUtils.px_to_metre(
            self.shared_data.reference_field_pts, 
            (self.tpl_h_px, self.tpl_w_px)
        )

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
            assert type(idx) == int and 0 <= idx < len(self.reference_field_meter)
            indices.append(idx)
        
        self.shared_data.reference_field_indices = indices
        self.logger.info("Template indices received. Now computing homography...")        
        return self.compute_homography()

    def health_check(self, H:np.ndarray) -> bool:

        # 1) normalize so H[2,2]==1
        if abs(H[2,2]) < 1e-8:
            self.logger.info("❌ health_check: H[2,2] is too small to normalize.")
            return False
        H = H / H[2,2]

        # 2) determinant must be far from zero
        det = float(np.linalg.det(H))
        if abs(det) < 1e-6:
            self.logger.info(f"❌ health_check: det(H) is too small: {det:.2e}")
            return False
        
        # 3) poject four image corners into field-met coords
        square = create_base_square((1, 3, self.h_px, self.w_px), as_tensor=False)
        square = square.reshape(-1, 1, 2) # (4,1,2)
        proj = cv2.perspectiveTransform(square, H).reshape(-1,2)

        # 4) how many lie inside [0,120]×[0,80]? require at least 3/4
        L, W = PitchConfig().length, PitchConfig().width
        inside = ((proj[:,0] >= -1e-3) & (proj[:,0] <= L+1e-3) &
                  (proj[:,1] >= -1e-3) & (proj[:,1] <= W+1e-3))
        good = int(inside.sum())
        if good < 3:
            self.logger.info(f"❌ health_check: only {good}/4 corners in field.")
            return False

        return True


    def compute_homography(self) -> bool:
        if len(self.shared_data.captured_video_pts) != 4 or len(self.shared_data.reference_field_indices) != 4:
            self.logger.info("Cannot compute homography. 4 points and 4 indices are required.")
            return False

        captured_pts = np.array(self.shared_data.captured_video_pts, dtype=np.float32)
        reference_pts = self.reference_field_meter[self.shared_data.reference_field_indices, :]


        # video_px to field_metre
        self.H_video2field, _ = cv2.findHomography(
            captured_pts, reference_pts, cv2.RANSAC)
        if self.H_video2field is None:
            self.phase= "done"
            self.logger.info("❌ Homography computation failed.")
            return False
        
        if not self.health_check(self.H_video2field):
            self.logger.info("❌ Homography failed health check.")
            self.phase= "done"
            return False
             
        self.H_field2video = np.linalg.inv(
            self.H_video2field + 1e-8 * np.eye(self.H_video2field.shape[0])
            )
        self.health_ok = True
        self.phase = "done"
        self.logger.info("Homography computed and passed health_check.")
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
            if not self.health_ok:
                self.logger.info("Homography failed health check. Not saving.")
            else:
                answer = input("Do you want to save this annotated frame? (Y/N): ").strip().lower()
                if answer == "y":
                    self.save_results(count=idx, output_dir=self.output_dir)
                    self.logger.info("Annotated frame saved.")
        else:
            self._generate_projection(self.visualizer)

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

        field_projected_pts = TransformUtils.metre_to_px(
            field_projected_pts, 
            (self.tpl_h_px, self.tpl_w_px)
        )

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
    import re
    import random
    import itertools
    frames_dir = Path("data/frames")
    output_dir = Path("data/annotated_homographies")
    output_dir.mkdir(parents=True, exist_ok=True)

    def next_annot_index() -> int:
        used = {
            int(m.group(1))
            for f in Path(output_dir).glob("annot_frame_*.png")
            if (m := re.match(r"annot_frame_(\d+)\.png$", f.name))
        }
        return next(i for i in itertools.count(1) if i not in used)


    if not any(frames_dir.iterdir()):
        import subprocess
        print(f"Frames folder is empty. Running frames extraction script...")
        subprocess.run(["python3", "-m", "src.model.perspect.get_frames"], check=True)
    else:
        print("Frames folder is not empty. Proceeding with annotation.")

    match_folders = sorted([f for f in frames_dir.iterdir() if f.is_dir()])
    
    MATCH = 2
    frame_files = list(match_folders[MATCH].glob("*.jpg"))

    n = 20
    random.seed(next_annot_index())
    
    sampled_frames = random.sample(frame_files, min(n, len(frame_files)))

    for frame_path in sampled_frames:
        print(f"\n Processing frame: {frame_path}")

        idx = next_annot_index()

        annotator = HomographyAnnotator(
            image_path=str(frame_path),
            shared_data=SharedAnnotations(),
            output_dir=output_dir
            )
        
        annotator.run(idx)

        print(f"Frame {frame_path.name} processed.")

    print("All frames processed.")


