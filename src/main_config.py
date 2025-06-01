# src/config/main_config.py
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
