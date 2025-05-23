#!/usr/bin/env python3
# map_rf2my.py

import cv2
import numpy as np
from pathlib import Path

from src.struct.shared_data import SharedAnnotations
from src.visual.field import PitchConfig, FieldVisualizer
from src.visual.visualizer import Visualizer
from src.visual.video import VideoFrame
from src.config import VisualizationConfig

# ── constants ───────────────────────────────────────────────────────────────
TRAIN_ROOT = Path("data/00--raw/football-field-detection.v15i.yolov5pytorch/train")
IMG_DIR    = TRAIN_ROOT / "images"
LBL_DIR    = TRAIN_ROOT / "labels"
# ─────────────────────────────────────────────────────────────────────────────

def load_keypoints(lbl_path: Path) -> np.ndarray:
    """Parse YOLO-KP label file into (32,3) array: (x_norm,y_norm,vis)."""
    vals = list(map(float, lbl_path.read_text().split()))
    return np.array(vals[5:], dtype=np.float32).reshape(-1,3)

class DummyProcess:
    def __init__(self, shared): self.shared_data = shared
    def is_done(self): return False
    def on_mouse_click(self, x, y): pass

def main():
    img_name = input("Enter image filename (e.g. foo.jpg): ").strip()
    img_path = IMG_DIR / img_name
    lbl_path = LBL_DIR / f"{img_path.stem}.txt"

    if not img_path.exists() or not lbl_path.exists():
        print("Missing image or label.")
        return

    img = cv2.imread(str(img_path))
    kp  = load_keypoints(lbl_path)  # (32,3)

    shared = SharedAnnotations()

    rf_numbered = []
    for (x,y,v) in kp:
        if v > 0:
            rf_numbered.append((x,y))
        else:
            rf_numbered.append((-1.0,-1.0))
    shared.numbered_video_points["rf"] = np.array(rf_numbered, dtype=np.float32)

    fv = FieldVisualizer(PitchConfig(), vis_config=VisualizationConfig())
    model_pts = fv._reference_model_pts()  # (33,2) in metres
    fld_numbered = []
    for u, v in model_pts:
        fld_numbered.append((u,v))
    shared.numbered_field_points["canon"] = np.array(fld_numbered, dtype=np.float32)

    proc = DummyProcess(shared)
    vis = Visualizer(
        field_config=PitchConfig(),
        frame=img,
        vis_config=VisualizationConfig(show_reference_points='points_text'),
        process=proc
    )
    vf = VideoFrame(frame_id=0, timestamp=0.0, image=img, detections=[])
    vis.update(vf)
    vis.show()

if __name__ == "__main__":
    main()


# 0a2d9b_2_3_png.rf.2b39030ff9f2e93a34aa9ca69abbd77c.jpg