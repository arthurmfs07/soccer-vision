from dataclasses import dataclass, field
from typing import *


from dataclasses import dataclass, field
from typing import Literal, Dict

@dataclass
class MainConfig:
    TRAIN_TYPE: Literal["square","points"] = "points"

    dataset_paths: Dict[str,str] = field(default_factory=lambda: {
        "square": "data/01--clean/roboflow",
        "points": "data/00--raw/football-field-detection.v15i.yolov5pytorch",
    })
    save_paths: Dict[str,str] = field(default_factory=lambda: {
        "square": "data/10--models/perspect_cnn_square_baseline_lowres.pth",
        "points": "data/10--models/perspect_cnn_points_heatmapsigma2_resnet18_coords_entropy0.1.pth",
    })

    yolo_paths: Dict[str,str] = field(default_factory=lambda: {
        "detect": "data/10--models/yolov8m_detect_imgsz320.pt",
        "keypoints": "runs/pose/new-yolov8m-pose-imgsz320/weights/best.pt"
    })

    # shared hyperparams
    BATCH_SIZE: int = 20
    LR:         float = 2e-4 # for square 1e-5
    PATIENCE:   int = 50
    DEVICE:     str = "cuda"

    @property
    def DATASET_PATH(self) -> str:
        return self.dataset_paths[self.TRAIN_TYPE]

    @property
    def SAVE_PATH(self) -> str:
        return self.save_paths[self.TRAIN_TYPE]


class BaseConfig:
    device: str = "cuda"
    batch_size: int = 32       # Size of a single inference batch
    target_fps: int = 8

    video_name: str = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
    # video_name: str = "JA0p0Bg9N1w.mp4"
    video_dir: str = "data/00--raw/videos"


@dataclass
class RealTimeConfig(BaseConfig):
    """Real-time inference config"""
    imgsz:           int   = 320       # inference image size
    max_buffer_size: int   = 200       # Visualization buffer size (in batches)
    skip_sec:        int   = 100*60    # Skip first 100 minutes of video
    target_fps:      int   = 10        # Target frames per second for visualization
    detect_conf:     float = 0.3       # Confidence threshold for YOLO
    annotation_gap: Literal[-1] = 100  # pixel gap between consecutive annotations


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configurations"""
    kp_conf_th:     float = 0.80   # threshold for keypoint model confidence  
    tc_conf_th:     float = 0.40   # TeamCluster threshld
    border_margin:  float = 0.01   # normalized amount considered border (filter out keypoints)


@dataclass
class HEConfig:
    """Homography estimation configurations."""
    smooth_alpha: float = 0.5      # [0..1] greater = more smooth
    max_err:      float = 0.01     # MSRE threshold in normalized units
    max_pts:      int   = 8        # maximum number of points to consider
    min_pts:      int   = 4        # min=4, filter frames with few pts
    min_inl:      int   = 6        # min_inl>=min_pts, filter homographies with few pts
    det_lo:       float = 5e-3     # filter if det(H) < det_lo
    det_hi:       float = 200      # filter if det(H) > det_hi
    thr_px:       float = 0.01     # RANSAC threshold in normalized coordinates



@dataclass
class VisualizationConfig(BaseConfig):
    """All display sizes; used only for denormalization at render time.
    All in (W,H) format"""
    video_disp_size: Tuple[int, int] = (640, 360)
    field_disp_size: Tuple[int, int] = (640, 360)
    # 'none'        -> don't draw them
    # 'points'      -> only black dots
    # 'points_text' -> black dots + index labels
    show_reference_points: Literal['none', 'points', 'points_text'] = 'points_text'

