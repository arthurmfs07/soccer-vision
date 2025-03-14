import cv2
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from src.logger import setup_logger

def get_color(color_name: str) -> Tuple[int, int, int]:
    """Converts simple color names to OpenCV BGR tuples."""
    mapping = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "green": (53, 125, 19),
        "blue": (255, 0, 0),
    }
    return mapping.get(color_name.lower(), (0, 0, 0))


def to_pixel(x: float, y: float, scale: float, offset_x: float, offset_y: float) -> Tuple[int, int]:
    """Converts field coordinates to image pixel coordinates."""
    return int((x + offset_x) * scale), int((y + offset_y) * scale)


@dataclass
class PitchConfig:
    length: int = 120
    width: int = 80
    origin_x: int = 0
    origin_y: int = 0
    line_color: str = "white"
    pitch_color: str = "green"
    linewidth: int = 1
    scale: float = 5.0



class FieldVisualizer:
    def __init__(self, config: PitchConfig):
        self.config = config
        self.logger = setup_logger("api.log")
        self.image = self._initialize_pitch()
        self.data_overlays: List[Dict] = []

    def _initialize_pitch(self) -> np.ndarray:
        """Creates the base pitch image."""
        c = self.config
        self.goal_offset = 2
        x_min = c.origin_x - self.goal_offset - 5
        x_max = c.origin_x + c.length + self.goal_offset + 5
        y_min = c.origin_y - 5
        y_max = c.origin_y + c.width + 5
        
        width_px = int((x_max - x_min) * c.scale)
        height_px = int((y_max - y_min) * c.scale)
        
        self.offset_x = -x_min
        self.offset_y = -y_min
        
        img = np.full((height_px, width_px, 3), 200, dtype=np.uint8)
        self._draw_pitch(img)
        return img

    def _draw_pitch(self, img: np.ndarray) -> None:
        """Draws the full pitch layout on the image."""
        c = self.config
        line_bgr = get_color(c.line_color)
        pitch_bgr = get_color(c.pitch_color)
        
        pt1 = to_pixel(c.origin_x, c.origin_y, c.scale, self.offset_x, self.offset_y)
        pt2 = to_pixel(c.origin_x + c.length, c.origin_y + c.width, c.scale, self.offset_x, self.offset_y)
        
        cv2.rectangle(img, pt1, pt2, pitch_bgr, thickness=-1)
        cv2.rectangle(img, pt1, pt2, line_bgr, thickness=c.linewidth)
        
        mid_x = c.origin_x + c.length / 2
        pt_top = to_pixel(mid_x, c.origin_y, c.scale, self.offset_x, self.offset_y)
        pt_bot = to_pixel(mid_x, c.origin_y + c.width, c.scale, self.offset_x, self.offset_y)
        cv2.line(img, pt_top, pt_bot, line_bgr, thickness=c.linewidth, lineType=cv2.LINE_AA)
        
        # Center circle
        center_px = to_pixel(mid_x, c.origin_y + c.width / 2, c.scale, self.offset_x, self.offset_y)
        radius_circle = int(10 * c.scale)
        cv2.circle(img, center_px, radius_circle, line_bgr, thickness=c.linewidth, lineType=cv2.LINE_AA)
        cv2.circle(img, center_px, 2, line_bgr, thickness=-1, lineType=cv2.LINE_AA)
        
        # Penalty areas
        for side in [0, c.length - 18]:
            pt1 = to_pixel(side, c.origin_y + (c.width - 44) / 2, c.scale, self.offset_x, self.offset_y)
            pt2 = to_pixel(side + 18, c.origin_y + (c.width + 44) / 2, c.scale, self.offset_x, self.offset_y)
            cv2.rectangle(img, pt1, pt2, line_bgr, thickness=c.linewidth)
        
        # Small areas (Goalkeeper box)
        for side in [0, c.length - 6]:
            pt1 = to_pixel(side, c.origin_y + (c.width - 20) / 2, c.scale, self.offset_x, self.offset_y)
            pt2 = to_pixel(side + 6, c.origin_y + (c.width + 20) / 2, c.scale, self.offset_x, self.offset_y)
            cv2.rectangle(img, pt1, pt2, line_bgr, thickness=c.linewidth)
        
        # Penalty marks
        for side in [12, c.length - 12]:
            penalty_spot = to_pixel(side, c.origin_y + c.width / 2, c.scale, self.offset_x, self.offset_y)
            cv2.circle(img, penalty_spot, 2, line_bgr, thickness=-1, lineType=cv2.LINE_AA)
        
        # Goals
        goal_width = 8
        for side in [c.origin_x - self.goal_offset, c.origin_x + c.length]:
            pt1 = to_pixel(side, c.origin_y + c.width / 2 - goal_width / 2, c.scale, self.offset_x, self.offset_y)
            pt2 = to_pixel(side + self.goal_offset, c.origin_y + c.width / 2 + goal_width / 2, c.scale, self.offset_x, self.offset_y)
            cv2.rectangle(img, pt1, pt2, line_bgr, thickness=c.linewidth)
        
    def add_overlay(self, overlay_data: Dict) -> None:
        """Stores data overlays for visualization."""
        self.data_overlays.append(overlay_data)

    def _draw_overlays(self, img: np.ndarray) -> None:
        """Placeholder method to draw overlays such as players' positions or field domination areas."""
        for overlay in self.data_overlays:
            pass  # Placeholder: Implement drawing logic based on overlay data

    def update(self) -> None:
        """Updates the pitch with overlays and refreshes the visualization."""
        self.image = self._initialize_pitch()
        self._draw_overlays(self.image)
        self.logger.info("Visualization updated with new overlays.")

    def show(self) -> None:
        """Displays the pitch visualization."""
        cv2.imshow("Pitch Visualizer", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_image(self) -> np.ndarray:
        """Return numpy array of the field visualization."""
        return self.image


if __name__ == "__main__":
    config = PitchConfig(scale=5, linewidth=1)
    field = FieldVisualizer(config)
    field.show()
