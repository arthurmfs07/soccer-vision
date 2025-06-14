import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Detection:
    """Stores detection results for a single frame."""
    boxes: np.ndarray       # Boundig boxes (x1, y1, x2, y2)
    confidences: np.ndarray # Confidence scores
    classes: np.ndarray     # Class IDs
