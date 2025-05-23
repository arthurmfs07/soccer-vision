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
        "square": "data/10--models/perspect_cnn10_square_baseline.pth",
        "points": "data/10--models/perspect_cnn10_points_baseline.pth",
    })

    # shared hyperparams
    BATCH_SIZE: int = 16
    LR:         float = 1e-5
    PATIENCE:   int = 30
    DEVICE:     str = "cuda"

    @property
    def DATASET_PATH(self) -> str:
        return self.dataset_paths[self.TRAIN_TYPE]

    @property
    def SAVE_PATH(self) -> str:
        return self.save_paths[self.TRAIN_TYPE]
