import time, warnings, queue, cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from src.logger import setup_logger
from src.visual.visualizer   import Visualizer
from src.process.annotator   import HomographyAnnotator
from src.config              import RealTimeConfig
from src.process.process     import Process
from src.struct.shared_data  import SharedAnnotations


class VisualizationProcess(Process):
    """
    Fetch frames from `buffer`, add annotations into SharedAnnotations,
    and render via the given `Visualizer`.
    """

    def __init__(
        self,
        buffer:        queue.Queue,
        visualizer:    Visualizer,
        config:        RealTimeConfig,
        shared_data:   SharedAnnotations,
        output_dir:    Optional[str] = None,
    ) -> None:
        self.logger      = setup_logger("visualization")
        self.buffer      = buffer
        self.vis         = visualizer
        self.shared_data = shared_data

        self.target_dt   = 1.0 / config.target_fps
        self.max_buf     = config.max_buffer_size
        self.annot_gap   = config.annotation_gap
        self.running     = True
        self.frame_id    = 0            # overall frame counter

        self.H_video2field = np.eye(3, dtype=np.float32)


        self.out_dir = Path(output_dir) if output_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.snap_idx = 0

    @staticmethod
    def _centre_foot(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) * 0.5, y2                       # (cx , bottom-y)

    def process_frames(self) -> None:
        self.logger.info("🎬  Visualisation loop started")
        while self.running:

            t0 = time.time()

            if self.buffer.qsize() < int(self.max_buf * 0.3):
                self.logger.debug(f"⏳ Buffer low ({self.buffer.qsize():d}/{self.max_buf})")
                time.sleep(0.5)
                continue

            try:
                vf = self.buffer.get(timeout=0.1)
            except queue.Empty:
                continue

            self.shared_data = vf.annotations

            self.vis.update(vf)
            self.vis.render()

            self.frame_id += 1
            if self.annot_gap > 0 and self.frame_id % self.annot_gap == 0:
                self._run_manual_annotation()

            dt = time.time() - t0
            if dt < self.target_dt:
                time.sleep(self.target_dt - dt)

        self.logger.info("✅  Visualization loop finished")

    def _run_manual_annotation(self) -> None:
        self.logger.info("⏸️  Pausing for manual annotation …")
        current_img = self.vis.video_vis.get_image().copy()

        annot = HomographyAnnotator(
            image_np=current_img,
            visualizer=self.vis,
            shared_data=self.shared_data,
        )
        annot.run()

        if self.out_dir:
            annot.save_results(self.snap_idx, self.out_dir)
            self.snap_idx += 1

        if self.shared_data.H_video2field is not None:
            self.H_video2field = self.shared_data.H_video2field
            self.logger.info("↺  Updated homography from manual annot.")

    def stop(self)            -> None: self.running = False
    def on_mouse_click(self,*_): pass
    def is_done(self)         -> bool : return not self.running
