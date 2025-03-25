from abc import ABC, abstractmethod
from typing import List, Tuple


class Process(ABC):
    """Abstract process class for interactive visualization tasks."""

    captured_video_pts:      List[Tuple[int, int]] = []
    sampled_video_pts:       List[Tuple[int, int]] = []
    video_detection_pts:     List[Tuple[int, int]] = []

    projected_field_pts:     List[Tuple[int, int]] = []
    projected_detection_pts: List[Tuple[int, int]] = []
    reference_field_indices: List[int] = []


    @abstractmethod
    def on_mouse_click(self, x: int, y: int) -> None:
        """Handles mouse click events in the visualizer."""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Checks whether the process is completed."""
        pass

    # @abstractmethod
    # def run(self) -> None:
    #     """Run the process"""
    #     pass