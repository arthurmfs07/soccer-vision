import cv2
import numpy as np
from typing import Optional, Dict

from src.logger import setup_logger
from src.config import VisualizationConfig
from src.visual.field import FieldVisualizer, PitchConfig
from src.visual.video import VideoVisualizer
from src.struct.shared_data import SharedAnnotations

class Visualizer:
    """Manages unified video + field visualization with purely normalized coordinates."""

    def __init__(
        self,
        field_config: PitchConfig,
        frame: np.ndarray,
        vis_config: VisualizationConfig,
        class_names: Dict[int, str] = None,
        process: Optional["Process"] = None
    ):
        self.logger = setup_logger("api.log")
        self.field_vis = FieldVisualizer(field_config)
        self.video_vis = VideoVisualizer(frame, class_names)
        self.process   = process

        self.video_size = frame.shape[:2]
        self.vis_cfg    = vis_config

        self.video_disp_sz = vis_config.video_disp_size
        self.field_disp_sz = vis_config.field_disp_size

        self.window_name = "Unified Visualization"
        self._dirty = True


    def get_image(self) -> np.ndarray:
        """Just a one‐off capture of the rendered view."""
        return self._generate_combined_view()


    def update(self, video_frame: "VideoFrame"):
        """Feed new YOLO detections into the video visualizer."""
        self.video_vis.update(video_frame)
        self._dirty = True


    def show(self) -> None:
        """Interactive loop—handles clicks and redraws until ‘q’ or process.is_done()."""

        def on_click(evt, x, y, flags, _):
            if evt == cv2.EVENT_LBUTTONDOWN and self.process:
                self.process.on_mouse_click(x, y)
                self._dirty = True

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, on_click)

        while True:
            if self._dirty:
                self._annotate_and_render()
                img = self._generate_combined_view()
                cv2.imshow(self.window_name, img)
                self._dirty = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (self.process and self.process.is_done()):
                break


    def render(self) -> None:
        """Non‐blocking single‐frame render (for real‐time pipelines)."""
        self._annotate_and_render()
        img = self._generate_combined_view()
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)


    def _annotate_and_render(self):
        """Clears old, adds fresh annotations based on normalized shared_data."""
        self.video_vis.clear_annotations()
        self.field_vis.clear_annotations()

        shared: SharedAnnotations = getattr(self.process, "shared_data", None)
        if shared is None:
            return

        w_vid, h_vid = self.video_disp_sz
        video_img = self.video_vis.get_image()
        w_fld, h_fld = self.field_disp_sz
        field_img = self.field_vis.get_image()

        for det in shared.yolo_detections:
            x1, y1, x2, y2 = det["bbox"]
            self.video_vis.frame.add_box(
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                width=(x2 - x1),
                height=(y2 - y1),
                color="yellow",
                thickness=0.003
            )

            if self.video_vis.class_names:
                cls = self.video_vis.class_names[det["class"]]
                y_off = 5.0 / h_vid
                self.video_vis.frame.add_text(
                    x=x1,
                    y=max(0.0, y1 - y_off),
                    text=cls,
                    color="yellow",
                    size=0.3
                )

        for color, pts, in shared.video_points.items():
            for (u, v) in pts:
                self.video_vis.frame.add_circle(
                    x=u, y=v, 
                    radius=0.01,
                    color=color, 
                    thickness=-1)

        for color, pts, in shared.field_points.items():
            for (u, v) in pts:
                self.field_vis.frame.add_circle(
                    x=u, y=v, 
                    radius=0.01,
                    color=color, 
                    thickness=-1)

        for color, pts, in shared.numbered_video_points.items():
            for idx, (u, v) in enumerate(pts):
                self.video_vis.frame.add_circle(
                    x=u, y=v, 
                    radius=0.01,
                    color=color, 
                    thickness=-1)
                self.video_vis.frame.add_text(
                    x=u, y=v,
                    text=str(idx),
                    color=color,
                    size=0.3
                )

        for color, pts, in shared.numbered_field_points.items():
            for idx, (u, v) in enumerate(pts):
                self.field_vis.frame.add_circle(
                    x=u, y=v, 
                    radius=0.01,
                    color=color, 
                    thickness=-1)
                self.field_vis.frame.add_text(
                    x=u, y=v,
                    text=str(idx),
                    color=color,
                    size=0.3
                )

        self.video_vis.frame.data.image = video_img
        self.field_vis.frame.data.image = field_img


    def _generate_combined_view(self) -> np.ndarray:
        """Stack the two denormalized & annotated views vertically."""
        vid = self.video_vis.get_image()
        fld = self.field_vis.get_image()

        if vid.shape[1] != fld.shape[1]:
            w = vid.shape[1]
            h = round(fld.shape[0] * (w / fld.shape[1]))
            fld = cv2.resize(fld, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.vstack((vid, fld))



if __name__ == "__main__":
    from src.visual.video import VideoFrame

    img_path = "data/00--raw/frames/match_3895113/t1_0.913.jpg"
    bgr      = cv2.imread(img_path)
    vis_cfg  = VisualizationConfig()

    shared = SharedAnnotations()

    shared.yolo_detections = [
        {"bbox": (0.1, 0.1, 0.3, 0.3), "class": 1},
        {"bbox": (0.6, 0.5, 0.9, 0.8), "class": 2},
    ]

    shared.video_points["red"]   = np.array([[0.1, 0.2], [0.25, 0.15]], dtype=np.float32)
    shared.video_points["green"] = np.array([[0.5, 0.2]], dtype=np.float32)
    shared.video_points["blue"]  = np.array([[0.8, 0.7], [0.7, 0.85]], dtype=np.float32)

    shared.field_points["red"]   = np.array([[0.2, 0.3]], dtype=np.float32)
    shared.field_points["green"] = np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float32)
    shared.field_points["blue"]  = np.array([[0.75, 0.25]], dtype=np.float32)

    class DummyProcess:
        def __init__(self, shared):  self.shared = shared
        def is_done(self):           return False
        def on_mouse_click(self, x, y): pass

    proc = DummyProcess(shared)

    class_names = {1: "TeamA", 2: "TeamB"}
    vis = Visualizer(
        PitchConfig(),      # your field config
        bgr,                # the test image
        vis_cfg,
        class_names=class_names,
        process=proc
    )

    vf = VideoFrame(
        frame_id=0,
        timestamp=0.0,
        image=bgr,
        detections=shared.yolo_detections
    )
    vis.update(vf)
    vis.show()
