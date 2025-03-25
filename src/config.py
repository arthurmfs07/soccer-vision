from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

@dataclass
class BaseConfig:
    device: str = "cuda"
    batch_size: int = 4       # Size of a single inference batch
    target_fps: int = 10

    video_name: str = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
    video_dir: str = "data/00--raw/videos"


@dataclass
class RealTimeConfig(BaseConfig):
    """Real-time inference config"""
    max_buffer_size: int = 100                   # Visualization buffer size (in batches)
    skip_sec: int = 100*60                       # Skip first 100 minutes of video
    target_fps: int = 10                         # Target frames per second for visualization
    yolo_conf : float = 0.5                      # Confidence threshold for YOLO
    annotation_gap: Literal["-1", "int"] = 100   # pixel gap between consecutive annotations

@dataclass
class YoloFinetuneConfig(BaseConfig):
    """YOLO finetuning config"""
    pass


@dataclass
class DataConfig(BaseConfig):
    """Data processing config"""
    width : int = 640
    height : int = 352
    shuffle: bool = False

@dataclass
class DownloadConfig(BaseConfig):
    """Download youtube video config"""
    numbers_to_download: int = 1

