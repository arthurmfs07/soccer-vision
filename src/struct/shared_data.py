import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class SharedAnnotations:

    # points presented in video
    captured_video_pts:      List[Tuple[int, int]] = field(default_factory=list)
    sampled_video_pts:       List[Tuple[int, int]] = field(default_factory=list)
    video_detection_pts:     List[Tuple[int, int]] = field(default_factory=list)

    # points presented in field
    projected_field_pts:     List[Tuple[int, int]] = field(default_factory=list)
    projected_detection_pts: List[Tuple[int, int]] = field(default_factory=list)
    reference_field_pts:     List[int] = field(default_factory=list)
    reference_field_indices: List[int] = field(default_factory=list)
