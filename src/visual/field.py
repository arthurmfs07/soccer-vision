import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.logger import setup_logger

from src.struct.frame import Frame
from src.struct.annotation import *
from src.struct.utils import get_color



@dataclass
class PitchConfig:
    """Configuration for a football pitch."""
    length:      float = 120.0
    width:       float = 80.0
    origin_x:    float = 0.0
    origin_y:    float = 0.0
    line_color:  str   = "white"
    pitch_color: tuple = "green"
    linewidth:   int   = 1
    margin:      float = 0.0
    goal_offset: float = 2.0
    draw_points: bool  = True
    bgc:         str   = "gray"


class FieldVisualizer:
    """
    Visualizes a soccer field by storing all drawing as Annotations
    in a single Frame object. Coordinates are in 'mode space',
    and are transformed at render time via Frame.scale + offsets
    """
        
    def __init__(
            self, 
            config: Optional[PitchConfig] = None,
            window_name: str = "Field Visualizer"
            ):
        super().__init__()
        self.config = config or PitchConfig()
        self.logger = setup_logger("field_visualizer.log")
        self.window_name = window_name
        
        self._initialize_params()
        
    def _initialize_params(self) -> Frame:
        """Creates the base pitch image in pixel space."""
        c = self.config
        self.x_min = c.origin_x - c.goal_offset - c.margin
        self.y_min = c.origin_y - c.margin
        self.x_max = c.origin_x + c.length + c.goal_offset + c.margin
        self.y_max = c.origin_y + c.width + c.margin

        raw_w = max(1, int(self.x_max - self.x_min))
        raw_h = max(1, int(self.y_max - self.y_min))

        base_image = np.full((raw_h, raw_w, 3), get_color(self.config.bgc), dtype=np.uint8)
        self.frame = Frame(base_image, metadata={"bgc": self.config.bgc})
        

        offset_x = 0
        offset_y = 0
        if self.x_min < 0:
            offset_x = int(-self.x_min)
        if self.y_min < 0:
            offset_y = int(-self.y_min)

        self.frame.update_padding((offset_y, 0, offset_x, 0))
        self._draw_pitch()

        if self.config.draw_points:
            self.draw_hardcoded_model_points()

        self.frame.set_static_checkpoint()

    
    def to_model_coords(self, points_px: np.ndarray) -> np.ndarray:
        """
        Converts pixel-space coordinates to model-space (field coordinates in meters).
        Args:
            points_px: np.ndarray of shape (N, 2) in pixel space.
        
        Returns:
            np.ndarray of shape (N, 2) in model space.
        """
        scale = self.frame._scale
        pad_top, _, pad_left, _ = self.frame._padding

        return np.array([
            ((x - pad_left) / scale, (y - pad_top) / scale)
            for x, y in points_px
        ], dtype=np.float32)

    def _draw_pitch(self):
        """
        Add all lines/shapes for the standard soccer field layout
        in model coordinates. The Frame will handle transformations.
        """
        c = self.config

        top_left_x = c.origin_x
        top_left_y = c.origin_y
        bot_right_x = c.origin_x + c.length
        bot_right_y = c.origin_y + c.width
        
        # Fill rectangle with pitch color
        self.frame.add_rect(
            x1=top_left_x, y1=top_left_y,
            x2=bot_right_x, y2=bot_right_y,
            color=c.pitch_color,
            thickness=-1  # fill
        )
        # Draw the border
        self.frame.add_rect(
            x1=top_left_x, y1=top_left_y,
            x2=bot_right_x, y2=bot_right_y,
            color=c.line_color,
            thickness=c.linewidth
        )

        # Middle line
        mid_x = c.origin_x + c.length / 2
        self.frame.add_line(
            x1=mid_x, y1=c.origin_y,
            x2=mid_x, y2=c.origin_y + c.width,
            color=c.line_color,
            thickness=c.linewidth
        )

        # Center circle
        center_x = mid_x
        center_y = c.origin_y + c.width / 2
        radius_m = 10.0  # radius in model units
        self.frame.add_circle(
            x=center_x, y=center_y,
            radius=radius_m,
            color=c.line_color,
            thickness=c.linewidth
        )
        # Center spot
        self.frame.add_circle(
            x=center_x, y=center_y,
            radius=1.5,
            color=c.line_color,
            thickness=-1  # filled
        )

        # Left penalty area
        top_left_pen_x = c.origin_x
        top_left_pen_y = c.origin_y + (c.width - 44) / 2
        bot_right_pen_x = c.origin_x + 18
        bot_right_pen_y = top_left_pen_y + 44
        self.frame.add_rect(
            x1=top_left_pen_x, y1=top_left_pen_y,
            x2=bot_right_pen_x, y2=bot_right_pen_y,
            color=c.line_color,
            thickness=c.linewidth
        )
        # Right penalty area
        top_left_pen2_x = c.origin_x + c.length - 18
        top_left_pen2_y = top_left_pen_y
        bot_right_pen2_x = c.origin_x + c.length
        bot_right_pen2_y = bot_right_pen_y
        self.frame.add_rect(
            x1=top_left_pen2_x, y1=top_left_pen2_y,
            x2=bot_right_pen2_x, y2=bot_right_pen2_y,
            color=c.line_color,
            thickness=c.linewidth
        )

        # Smaller (goal) areas
        top_left_goal_x = c.origin_x
        top_left_goal_y = c.origin_y + (c.width - 20) / 2
        bot_right_goal_x = c.origin_x + 6
        bot_right_goal_y = top_left_goal_y + 20
        self.frame.add_rect(
            x1=top_left_goal_x, y1=top_left_goal_y,
            x2=bot_right_goal_x, y2=bot_right_goal_y,
            color=c.line_color,
            thickness=c.linewidth
        )
        # Opposite side
        top_left_goal2_x = c.origin_x + c.length - 6
        top_left_goal2_y = top_left_goal_y
        bot_right_goal2_x = c.origin_x + c.length
        bot_right_goal2_y = bot_right_goal_y
        self.frame.add_rect(
            x1=top_left_goal2_x, y1=top_left_goal2_y,
            x2=bot_right_goal2_x, y2=bot_right_goal2_y,
            color=c.line_color,
            thickness=c.linewidth
        )

        # Penalty spots
        left_pen_spot_x = c.origin_x + 12
        pen_spot_y = c.origin_y + c.width / 2
        right_pen_spot_x = c.origin_x + c.length - 12
        for spot_x in [left_pen_spot_x, right_pen_spot_x]:
            self.frame.add_circle(
                x=spot_x, y=pen_spot_y,
                radius=1.5,
                color=c.line_color,
                thickness=-1
            )

        # Goals (just outside the pitch)
        goal_w = 8.0
        left_goal_x1 = c.origin_x - c.goal_offset
        left_goal_y1 = c.origin_y + c.width/2 - goal_w/2
        left_goal_x2 = c.origin_x
        left_goal_y2 = left_goal_y1 + goal_w
        self.frame.add_rect(
            x1=left_goal_x1,  y1=left_goal_y1,
            x2=left_goal_x2,  y2=left_goal_y2,
            color=c.line_color,
            thickness=c.linewidth
        )

        right_goal_x1 = c.origin_x + c.length
        right_goal_y1 = left_goal_y1
        right_goal_x2 = right_goal_x1 + c.goal_offset
        right_goal_y2 = left_goal_y1 + goal_w
        self.frame.add_rect(
            x1=right_goal_x1,  y1=right_goal_y1,
            x2=right_goal_x2,  y2=right_goal_y2,
            color=c.line_color,
            thickness=c.linewidth
        )

    def get_hardcoded_model_points(self) -> np.ndarray:
        """Returns a set of reference points in model space (original field coordinates)."""
        c = self.config
        
        # Define key points in model space
        raw_points = [
            # Corners
            (c.origin_x, c.origin_y),  # 0: Top-left
            (c.origin_x + c.length, c.origin_y),  # 1: Top-right
            (c.origin_x, c.origin_y + c.width),  # 2: Bottom-left
            (c.origin_x + c.length, c.origin_y + c.width),  # 3: Bottom-right
            
            # Center
            (c.origin_x + c.length / 2, c.origin_y + c.width / 2),  # 4: Center spot
            
            # Penalty spots
            (c.origin_x + 12, c.origin_y + c.width / 2),  # 5: Left penalty spot
            (c.origin_x + c.length - 12, c.origin_y + c.width / 2),  # 6: Right penalty spot
            
            # Left penalty area corners
            (c.origin_x, c.origin_y + (c.width - 44) / 2),  # 7: Top left of left penalty area
            (c.origin_x, c.origin_y + (c.width + 44) / 2),  # 8: Bottom left of left penalty area
            (c.origin_x + 18, c.origin_y + (c.width - 44) / 2),  # 9: Top right of left penalty area
            (c.origin_x + 18, c.origin_y + (c.width + 44) / 2),  # 10: Bottom right of left penalty area
            
            # Right penalty area corners
            (c.origin_x + c.length, c.origin_y + (c.width - 44) / 2),  # 11: Top right of right penalty area
            (c.origin_x + c.length, c.origin_y + (c.width + 44) / 2),  # 12: Bottom right of right penalty area
            (c.origin_x + c.length - 18, c.origin_y + (c.width - 44) / 2),  # 13: Top left of right penalty area
            (c.origin_x + c.length - 18, c.origin_y + (c.width + 44) / 2),  # 14: Bottom left of right penalty area
            
            # Left goal area corners
            (c.origin_x, c.origin_y + (c.width - 20) / 2),  # 15: Top left of left goal area
            (c.origin_x, c.origin_y + (c.width + 20) / 2),  # 16: Bottom left of left goal area
            (c.origin_x + 6, c.origin_y + (c.width - 20) / 2),  # 17: Top right of left goal area
            (c.origin_x + 6, c.origin_y + (c.width + 20) / 2),  # 18: Bottom right of left goal area
            
            # Right goal area corners
            (c.origin_x + c.length, c.origin_y + (c.width - 20) / 2),  # 19: Top right of right goal area
            (c.origin_x + c.length, c.origin_y + (c.width + 20) / 2),  # 20: Bottom right of right goal area
            (c.origin_x + c.length - 6, c.origin_y + (c.width - 20) / 2),  # 21: Top left of right goal area
            (c.origin_x + c.length - 6, c.origin_y + (c.width + 20) / 2),  # 22: Bottom left of right goal area
            
            # Midfield line endpoints
            (c.origin_x + c.length / 2, c.origin_y),  # 23: Top of midfield line
            (c.origin_x + c.length / 2, c.origin_y + c.width),  # 24: Bottom of midfield line
            
            # Center circle top & bottom
            (c.origin_x + c.length / 2, c.origin_y + c.width / 2 - 10),  # 25: Top of center circle
            (c.origin_x + c.length / 2, c.origin_y + c.width / 2 + 10),  # 26: Bottom of center circle
            
            # Additional points
            (c.origin_x + 18, c.origin_y + 32),  # 27
            (c.origin_x + 18, c.origin_y + 48),  # 28
            (c.origin_x + c.length - 18, c.origin_y + 32),  # 29
            (c.origin_x + c.length - 18, c.origin_y + 48),  # 30
            (c.origin_x + 50, c.origin_y + 40),  # 31
            (c.origin_x + 70, c.origin_y + 40),  # 32
        ]
        
        return np.array(raw_points, dtype=np.float32)
    
    def draw_hardcoded_model_points(self):
        """Draws the hardcoded reference points on the provided image."""
        model_pts = self.get_hardcoded_model_points()
        for i, (mx, my) in enumerate(model_pts):
            self.frame.add_circle(
                x=mx, y=my, radius=1.2,
                color="black", thickness=-1
            )
            self.frame.add_text(
                x=mx + 2, y=my + 2,
                text=str(i),
                color="white",
                size=0.4
            )

    def get_template_pixel_points(self) -> np.ndarray:
        """Get pixel space set of hardcoded points."""
        model_pts = self.get_hardcoded_model_points()

        out = []
        top_padding  = self.frame._padding[0]
        left_padding = self.frame._padding[2]
        current_scale = self.frame._scale

        for mx, my in model_pts:
            px = mx * current_scale + left_padding
            py = my * current_scale + top_padding
            out.append((px, py))

        return np.array(out, dtype=np.float32)
    
    
    def clear_annotations(self) -> None:
        if isinstance(self.frame, Frame):
            self.frame.clear_annotations()        

    def show(self):
        """Standalone test."""
        cv2.imshow(self.window_name, self.frame.rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, filepath: str):
        """Save the final field to an image file."""
        cv2.imwrite(filepath, self.frame.rendered)
        self.logger.info(f"Pitch visualization saved to {filepath}")

    def get_image(self) -> np.ndarray:
        """Return a copy of the final field as a numpy array (BGR)."""
        return self.frame.rendered.copy()


if __name__ == "__main__":
    field = FieldVisualizer()
    # Add a test point in the center
    c = field.config
    center_x = c.origin_x + c.length / 2
    center_y = c.origin_y + c.width / 2
    field.frame.add_point(center_x, center_y, "blue", radius=4)
    
    field.draw_hardcoded_model_points()
    # Show on screen
    field.show()