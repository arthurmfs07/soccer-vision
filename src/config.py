from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class BaseConfig:
    device: str = "cuda"

@dataclass
class VideoConfig(BaseConfig):
    fps: int = 10 # 10 frames per second
    pass

@dataclass
class VisConfig(BaseConfig):
    """Online visualizer config"""
    pass

@dataclass
class HomoConfig(BaseConfig):
    """Homography transformation model config"""
    pass

@dataclass
class MOTConfig(BaseConfig):
    """Multi-Object Tracking model config"""
    pass

@dataclass
class DownloadConfig(BaseConfig):
    """Download youtube video config"""
    pass
