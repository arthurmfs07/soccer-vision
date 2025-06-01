# file: src/visual/field.py

# NOTE: all metric sizes now follow IFAB Law-1 exactly.


import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.logger import setup_logger
from src.config import VisualizationConfig
from src.struct.frame import Frame
from src.struct.annotation import LineAnnotation, CircleAnnotation, RectTLAnnotation
from src.struct.utils import get_color


@dataclass
class PitchConfig:
    length:      float = 105.0          # m  – goal-line to goal-line
    width:       float =  68.0          # m  – touch-line to touch-line
    origin_x:    float =   0.0
    origin_y:    float =   0.0
    line_color:  str   = "white"
    pitch_color: str   = "green"
    linewidth:   float =   0.12         # m  – 12 cm (max)
    margin:      float =   0.0          # extra space around model
    goal_offset: float =   2.0          # draw goals outside field
    bgc:         str   = "gray"
    draw_points: bool  =  True          # draw the 33 reference dots


class FieldVisualizer:
    """
    Draw a FIFA-standard pitch in normalised [0,1] coordinates and add
    33 fixed reference points for homography / calibration.
    """

    # canonical IFAB sizes (all metres)
    PA_D, PA_W = 16.5, 40.32            # penalty area depth / width
    GA_D, GA_W =  5.5, 18.32            # goal area    depth / width
    SPOT_D     = 11.0                   # penalty mark distance
    CIRCLE_R   =  9.15                  # centre circle + “D” radius

    def __init__(
        self,
        config:      Optional[PitchConfig]         = None,
        vis_config:  Optional[VisualizationConfig] = None,
        window_name: str                           = "Field Visualizer",
        ):
        self.cfg     = config or PitchConfig()
        self.vis_cfg = vis_config or VisualizationConfig()
        self.window  = window_name
        self.log     = setup_logger("field_visualizer")

        c = self.cfg
        self.x_min   = c.origin_x - c.goal_offset - c.margin
        self.y_min   = c.origin_y - c.margin
        self.x_max   = c.origin_x + c.length + c.goal_offset + c.margin
        self.y_max   = c.origin_y + c.width  + c.margin
        self.w_m     = self.x_max - self.x_min         # total model width  (m)
        self.h_m     = self.y_max - self.y_min         # total model height (m)

        blank = np.full((1, 1, 3), get_color(c.bgc), dtype=np.uint8)
        self.frame = Frame(blank)

        self._draw_pitch()
        mode = self.vis_cfg.show_reference_points
        if mode in ('points', 'points_text'):
            self._draw_reference_points(with_text=(mode == 'points_text'))

        self._static_count = len(self.frame.data.annotations)

    def _to_norm(self, x: float, y: float) -> Tuple[float, float]:
        """Convert model-space metres → normalised [0,1]."""
        return ((x - self.x_min) / self.w_m,
                (y - self.y_min) / self.h_m)

    def _add_arc(
        self,
        centre_u: float,
        centre_v: float,
        radius_m: float,
        ang0_deg: float,
        ang1_deg: float,
        colour: str,
        t_norm: float,
        seg: int = 64,
    ):
        """Ellipse-aware arc (x- & y-radii scale independently)."""
        rad_u = radius_m / self.w_m
        rad_v = radius_m / self.h_m
        angs  = np.linspace(np.deg2rad(ang0_deg), np.deg2rad(ang1_deg), seg+1)

        pts = [(centre_u + rad_u*np.cos(a),
                centre_v + rad_v*np.sin(a))
               for a in angs]

        for (u1, v1), (u2, v2) in zip(pts, pts[1:]):
            self.frame.add_line(u1, v1, u2, v2, colour, t_norm)

    def _draw_pitch(self) -> None:
        c      = self.cfg
        t_norm = c.linewidth / self.w_m   # line-width in [0,1] units

        u1, v1 = self._to_norm(c.origin_x,            c.origin_y)
        u2, v2 = self._to_norm(c.origin_x + c.length, c.origin_y + c.width)
        self.frame.add_rect(u1, v1, u2, v2, c.pitch_color, -1)
        self.frame.add_rect(u1, v1, u2, v2, c.line_color,  t_norm)

        midx = c.origin_x + c.length/2
        self.frame.add_line(*self._to_norm(midx, c.origin_y),
                            *self._to_norm(midx, c.origin_y + c.width),
                            c.line_color, t_norm)

        cu, cv = self._to_norm(midx, c.origin_y + c.width/2)
        self._add_arc(cu, cv, self.CIRCLE_R, 0, 360, c.line_color, t_norm)

        self.frame.add_circle(cu, cv, 1.5/self.w_m, c.line_color, -1)

        for side in (-1, 1):                         # -1 = left, +1 = right
            gx   = c.origin_x + (0 if side == -1 else c.length)
            mul  = 1 if side == -1 else -1

            # penalty area rectangle
            self.frame.add_rect(
                *self._to_norm(gx + mul*self.PA_D,
                               c.origin_y + (c.width - self.PA_W)/2),
                *self._to_norm(gx,
                               c.origin_y + (c.width + self.PA_W)/2),
                c.line_color, t_norm
            )

            # goal area rectangle
            self.frame.add_rect(
                *self._to_norm(gx + mul*self.GA_D,
                               c.origin_y + (c.width - self.GA_W)/2),
                *self._to_norm(gx,
                               c.origin_y + (c.width + self.GA_W)/2),
                c.line_color, t_norm
            )

            # penalty mark
            pm_u, pm_v = self._to_norm(gx + mul*self.SPOT_D,
                                       c.origin_y + c.width/2)
            self.frame.add_circle(pm_u, pm_v, 1.5/self.w_m, c.line_color, -1)

            # “D” arc (joins penalty area)
            ang0, ang1 = (-53, 53) if side == -1 else (127, 233)
            self._add_arc(pm_u, pm_v, self.CIRCLE_R,
                          ang0, ang1, c.line_color, t_norm)

    def _reference_model_pts(self) -> np.ndarray:
        """Return 33 reference points in model-space (norm)."""
        import math
        c = self.cfg
        mid_y   = c.origin_y + c.width/2
        left_x  = c.origin_x + self.PA_D
        right_x = c.origin_x + c.length - self.PA_D
        dy = math.sqrt(self.CIRCLE_R**2 - (self.PA_D - self.SPOT_D)**2)  # ≈ 7.312 m
        cx = c.origin_x + c.length/2          # pitch centre-x

        meter_pts: List[Tuple[float, float]] = [
            # 0–3  corners
            (c.origin_x,            c.origin_y),
            (c.origin_x+c.length,   c.origin_y),
            (c.origin_x,            c.origin_y+c.width),
            (c.origin_x+c.length,   c.origin_y+c.width),
            # 4    centre
            # (cx, mid_y),
            # 5–6  penalty spots
            (c.origin_x + self.SPOT_D,            mid_y),
            (c.origin_x + c.length - self.SPOT_D, mid_y),

            # 7–10  left PA
            (c.origin_x,             mid_y-self.PA_W/2),
            (c.origin_x,             mid_y+self.PA_W/2),
            (left_x,                 mid_y-self.PA_W/2),
            (left_x,                 mid_y+self.PA_W/2),

            # 11–14 right PA
            (c.origin_x+c.length,             mid_y-self.PA_W/2),
            (c.origin_x+c.length,             mid_y+self.PA_W/2),
            (right_x,                         mid_y-self.PA_W/2),
            (right_x,                         mid_y+self.PA_W/2),

            # 15–18 left GA
            (c.origin_x,             mid_y-self.GA_W/2),
            (c.origin_x,             mid_y+self.GA_W/2),
            (c.origin_x+self.GA_D,   mid_y-self.GA_W/2),
            (c.origin_x+self.GA_D,   mid_y+self.GA_W/2),

            # 19–22 right GA
            (c.origin_x+c.length,             mid_y-self.GA_W/2),
            (c.origin_x+c.length,             mid_y+self.GA_W/2),
            (c.origin_x+c.length-self.GA_D,   mid_y-self.GA_W/2),
            (c.origin_x+c.length-self.GA_D,   mid_y+self.GA_W/2),

            # 23–24 midfield line ends
            (cx, c.origin_y),
            (cx, c.origin_y+c.width),

            # 25–26 circle top / bottom
            (cx, mid_y - self.CIRCLE_R),
            (cx, mid_y + self.CIRCLE_R),

            # 27–30 “D” arc endpoints
            (left_x,  mid_y - dy),
            (left_x,  mid_y + dy),
            (right_x, mid_y - dy),
            (right_x, mid_y + dy),

            # 31–32 NEW: circle left / right extrema
            (cx - self.CIRCLE_R, mid_y),       # leftmost on circle
            (cx + self.CIRCLE_R, mid_y),       # rightmost on circle

        ]

        norm_pts = [self._to_norm(x_m, y_m) for x_m, y_m in meter_pts]
        return np.array(norm_pts, dtype=np.float32)


    def _draw_reference_points(self, with_text: bool = False) -> None:
        for idx, (u, v) in enumerate(self._reference_model_pts()):
            self.frame.add_circle(u, v, 1.2/self.w_m, color="black", thickness=-1)
            if with_text:
                self.frame.add_text(x=u, y=v,
                                    text=str(idx),
                                    color="black",
                                    size=0.2)

    def clear_annotations(self) -> None:
        """Remove dynamic annotations but keep static pitch + dots."""
        self.frame.data.annotations = self.frame.data.annotations[: self._static_count]

    # ── rendering (adds 2 % gray frame) ──────────────────────────
    def get_image(self) -> np.ndarray:
        W_px, H_px = self.vis_cfg.field_disp_size
        pad        = 0.02
        W_i        = round(W_px * (1 - 2*pad))
        H_i        = round(H_px * (1 - 2*pad))
        inner      = self.frame.render((W_i, H_i))

        canvas = np.full((H_px, W_px, 3), get_color(self.cfg.bgc), dtype=np.uint8)
        x0, y0 = round(W_px * pad), round(H_px * pad)
        canvas[y0:y0+H_i, x0:x0+W_i] = inner
        return canvas

    def show(self) -> None:
        cv2.imshow(self.window, self.get_image())
        cv2.waitKey(0)
        cv2.destroyWindow(self.window)


if __name__ == "__main__":
    FieldVisualizer().show()