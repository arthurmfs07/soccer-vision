# file: src/visual/video.py

import cv2
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from src.logger import setup_logger

from src.struct.frame import Frame
from src.struct.detection import Detection
from src.config import VisualizationConfig


@dataclass
class VideoFrame:
    """Encapsulate a single video frame with raw image, timestamp, ID, and detections."""
    frame_id:   int
    timestamp:  float
    image:      np.ndarray            # H×W×3, RGB or BGR uint8
    detections: List[Detection]      
    H:          Optional[np.ndarray] = None


class VideoVisualizer:
    """
    Wraps video frames into our Frame+Annotation system.
    - Converts raw pixel detections into normalized coords [0..1].
    - Keeps a Frame holding the raw image + normalized Annotation objects.
    - Renders by resizing at draw-time (no mid-pipeline scaling).
    """

    def __init__(
        self,
        initial_frame: np.ndarray,
        class_names:    Dict[int, str] = None,
        color_map:      Dict[int, str] = None,
        window_name:    str = "Video Visualizer"
    ):
        self.logger       = setup_logger("api.log")
        self.class_names  = class_names or {}
        self.vis_cfg = VisualizationConfig()

        # e.g. {0:"white",1:"blue",2:"green",3:"orange"}
        self.color_map    = color_map or {
            0: "white",      # ball
            1: "blue",       # goalkeeper
            2: "green",      # player
            3: "orange",     # referee
        }
        self.window_name  = window_name

        raw = self._prepare_frame(initial_frame)
        self.frame = Frame(raw)  # no annotations yet


    def update(
        self,
        video_frame: VideoFrame
    ) -> None:
        """
        Replace self.frame with a fresh Frame containing:
          - the raw image
          - BoxAnnotations for each detection, in normalized coords
        """
        img = self._prepare_frame(video_frame.image)
        H_px, W_px = img.shape[:2]
        f = Frame(img)

        for det in video_frame.detections:
            for box, cls, in zip(det.boxes, det.classes):
                x1, y1, x2, y2 = box
                if max(x1, y1, x2, y2) <= 1.0: # decide whether already normalized
                    x1_n, y1_n, x2_n, y2_n = x1, y1, x2, y2
                else:
                    x1_n, y1_n = x1 / W_px, y1 / H_px
                    x2_n, y2_n = x2 / W_px, y2 / H_px

                cx = ((x1_n + x2_n) / 2)
                cy = ((y1_n + y2_n) / 2)
                w_n = (x2_n - x1_n)
                h_n = (y2_n - y1_n)

                label          = self.class_names.get(cls, str(cls))
                color          = self.color_map.get(cls, "red")

                f.add_box(
                    x=cx,
                    y=cy,
                    width=w_n,
                    height=h_n,
                    label=label,
                    color=color
                )

        self.frame = f


    def clear_annotations(self) -> None:
        """Remove all annotations (kept for compatibility)."""
        self.frame.clear_annotations()


    def get_image(self) -> np.ndarray:
        """
        Render and return the annotated frame at disp_size=(H_px, W_px).
        """
        return self.frame.render(self.vis_cfg.video_disp_size)


    def show(self) -> None:
        """
        Render at disp_size and display in a cv2 window.
        """
        img = self.get_image()
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)


    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Ensure we have a H×W×3 uint8 BGR image.
        Accepts torch.Tensor or RGB numpy too.
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)

        # if it's RGB, convert to BGR for OpenCV
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame




if __name__ == "__main__":
    import numpy as _np
    dummy = _np.zeros((480,640,3),dtype=_np.uint8)
    from src.visual.video import VideoFrame
    vis = VideoVisualizer(dummy)
    # simulate one detection
    bbox = (160,120,480,360)
    vf = VideoFrame(frame_id=0, timestamp=0.0, image=dummy, detections=[{"bbox":bbox,"class":2}])
    vis.update(vf)
    vis.show((360,640))