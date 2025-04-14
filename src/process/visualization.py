# real time inference handling
import time
import cv2
import queue
import numpy as np
from pathlib import Path
from typing import Tuple, List, Literal, Optional

from src.process.annotator import HomographyAnnotator
from src.visual.visualizer import Visualizer
from src.config import RealTimeConfig
from src.process.process import Process
from src.struct.frame import Frame
from src.struct.shared_data import SharedAnnotations



class VisualizationProcess(Process):
    """Retrieves results from the buffer and visualizes in real-time."""
    
    def __init__(
            self, 
            buffer:        queue.Queue, 
            visualizer:    Visualizer, 
            config:        RealTimeConfig,
            H_video2field: np.ndarray = None,
            shared_data: SharedAnnotations = SharedAnnotations(),
            output_dir:  Optional[str] = None
            ):
        self.buffer = buffer
        self.visualizer = visualizer
        self.max_buffer_size = config.max_buffer_size
        self.batch_size = config.batch_size
        self.target_fps = config.target_fps
        self.frame_interval = 1.0 / config.target_fps
        self.running = True
        self.annotation_gap = config.annotation_gap
        self.frame_counter = 0
        self.H_video2field = H_video2field
        self.shared_data = shared_data

        self.output_dir = Path(output_dir)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.annot_count = 0

    def process_frames(self):
        """Continuously fetches frames and renders at 1 sec per sec."""
        start = True

        while self.running:

            start_time = time.time()

            required_size = int(self.max_buffer_size * (start*0.3 + 0.5))
            if self.buffer.qsize() < required_size:
                print(f"⏳ Buffer low ({self.buffer.qsize()}/{self.max_buffer_size}), waiting for more frames...")
                time.sleep(2)
                continue
            try:
                video_frame = self.buffer.get(timeout=0.1) 
                self.visualizer.update(video_frame)

            except queue.Empty:
                print("⏳ Buffer is empty, waiting for frames...")
                time.sleep(0.1)
                continue

            if self.H_video2field is not None:
                projected = []
                for det in video_frame.detections:
                    for det_box in det.boxes:
                        foot_pt = self._get_foot_pt(det_box)
                        projected = self._project_foot(foot_pt)
                        self.shared_data.projected_detection_pts.extend(projected)

            self.frame_counter += 1

            if self.annotation_gap > 0 and (self.frame_counter % self.annotation_gap == 0):
                print("⏸️  Pausing video process for annotation...")
                current_img = self.visualizer.video_visualizer.get_image().copy()
                annotator = HomographyAnnotator(
                    image_np=current_img, 
                    visualizer=self.visualizer, 
                    shared_data=self.shared_data,
                    )
                annotator.run() 

                if self.output_dir:
                    annotator.save_results(count=self.annot_count, output_dir=self.output_dir)
                    self.annot_count += 1

                H = self.shared_data.H_video2field
                if H is not None:
                    print("✅ Annotation completed. Updating H matrix.")
                    self.H_video2field = H 
                else:
                    print("⚠️  Annotation did not produce a valid homography.")
                    
            self.visualizer.render()

            elapsed_time = time.time() - start_time
            remaining_time = self.frame_interval - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

        print("✅ Visualization Process Finished")

    def _get_foot_pt(self, det_box: Tuple[int, int, int ,int]):
        x1, y1, x2, y2 = det_box
        foot_pt = np.array([[[ (x1+x2)/2, y2 ]]], dtype=np.float32)
        return foot_pt

    def _project_foot(self, foot_pt: np.ndarray) -> List[Tuple]:
        """
        foot_pt: video_pixel coordinate
        projected: field_model coordinate
        """

        if self.H_video2field is None:
            return []
        
        # print(f"foot_pt: {foot_pt}")

        projected = cv2.perspectiveTransform(
            foot_pt.reshape(-1, 1, 2), 
            self.H_video2field,
            ).reshape(-1, 2)

        return [tuple(projected[0])]


    def stop(self):
        """Stops the visualization process."""
        self.running = False

    def on_mouse_click(self, x: int, y: int) -> None:
        """Handles mouse click events in the visualizer."""
        pass

    def is_done(self) -> bool:
        """Checks whether the process is completed."""
        return False
