import numpy as np
from typing import Literal, Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class SharedAnnotations:


    # maps color name -> (Nx2) array of normalized [0..1] points
    video_points: Dict[Literal["red", "yellow", "blue"], np.ndarray] = field(
        default_factory=lambda: defaultdict(
            lambda: np.zeros((0, 2), dtype=np.float32)
        )
    )

    field_points: Dict[Literal["red", "yellow", "blue"], np.ndarray] = field(
        default_factory=lambda: defaultdict(
            lambda: np.zeros((0, 2), dtype=np.float32)
        )
    )

    numbered_video_points: Dict[Literal["red", "yellow", "blue"], np.ndarray] = field(
        default_factory=lambda: defaultdict(
            lambda: np.zeros((0, 2), dtype=np.float32)
        )
    )

    numbered_field_points: Dict[Literal["red", "yellow", "blue"], np.ndarray] = field(
        default_factory=lambda: defaultdict(
            lambda: np.zeros((0, 2), dtype=np.float32)
        )
    )

    yolo_detections: List[Dict[str, Any]] = field(default_factory=list)

    H_video2field: np.ndarray = np.eye(3, dtype=np.float32)
    H_field2video: np.ndarray = np.eye(3, dtype=np.float32)
