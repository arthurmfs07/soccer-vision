from dataclasses import dataclass
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np

# Replace with your actual logger import
from src.logger import setup_logger

@dataclass
class PitchConfig:
    pitch_length: int = 120
    pitch_width: int = 80
    origin_x: int = 0
    origin_y: int = 0
    line_color: str = "black"
    pitch_color: str = "white"
    linewidth: int = 1
    alpha: float = 1


class HomographyPitchDrawer:
    """Draws a StatsBomb-style soccer field with reference points for homography."""

    def __init__(self, config: Optional[PitchConfig] = None) -> None:
        """Initialize the pitch drawer with a config and logger."""
        self.logger = setup_logger("api.log")
        if config is None:
            config = PitchConfig()
        self.config = config

        # If you want to force a specific backend (optional):
        # matplotlib.use("TkAgg")

        self.figure, self.ax = plt.subplots(figsize=(16, 9))
        self.logger.info("HomographyPitchDrawer initialized.")

    def draw_pitch(self) -> None:
        """Draw the soccer field and reference points."""
        self._draw_statsbomb_pitch()
        self._draw_reference_points()
        self.logger.info("Pitch drawing completed.")

    def get_numpy_representation(self) -> np.ndarray:
        """Return a numpy array of the current pitch figure."""
        self.figure.canvas.draw()
        width, height = self.figure.canvas.get_width_height()

        # 'FigureCanvasTkAgg' has only tostring_argb(), not tostring_rgb()
        argb_data = self.figure.canvas.tostring_argb()
        buffer = np.frombuffer(argb_data, dtype=np.uint8).reshape((height, width, 4))

        # Convert ARGB -> RGB by dropping alpha channel
        rgb_buffer = buffer[..., [1, 2, 3]]

        self.logger.info(f"Numpy representation has shape {rgb_buffer.shape}.")
        return rgb_buffer

    def show_figure(self) -> None:
        """Display the current pitch figure."""
        plt.show()

    def save_figure(self, filepath: str) -> None:
        """Save the current pitch figure to a file."""
        try:
            self.figure.savefig(filepath, transparent=True, bbox_inches='tight')
            self.logger.info(f"Figure saved to {filepath}.")
        except OSError as e:
            self.logger.error(f"Error saving figure to {filepath}: {str(e)}")

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image as a numpy array and log its shape."""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            self.logger.info(f"Loaded image '{image_path}' with shape {img_array.shape}.")
            return img_array
        except FileNotFoundError:
            self.logger.error(f"Image not found: {image_path}")
        except OSError as e:
            self.logger.error(f"Error loading image '{image_path}': {str(e)}")
        return np.array([])

    def _draw_statsbomb_pitch(self) -> None:
        """Draw the main pitch lines in the figure."""
        cfg = self.config
        ax = self.ax

        pitch = patches.Rectangle(
            (cfg.origin_x, cfg.origin_y),
            cfg.pitch_length,
            cfg.pitch_width,
            edgecolor=cfg.line_color,
            facecolor=cfg.pitch_color,
            lw=cfg.linewidth,
            alpha=cfg.alpha
        )
        ax.add_patch(pitch)

        ax.plot([cfg.origin_x + cfg.pitch_length / 2, cfg.origin_x + cfg.pitch_length / 2],
                [cfg.origin_y, cfg.origin_y + cfg.pitch_width],
                color=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha)

        center_x = cfg.origin_x + cfg.pitch_length / 2
        center_y = cfg.origin_y + cfg.pitch_width / 2
        center_circle = patches.Circle(
            (center_x, center_y),
            radius=10, fill=False, edgecolor=cfg.line_color,
            lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(center_circle)
        ax.plot(center_x, center_y, marker='o', markersize=2,
                color=cfg.line_color, alpha=cfg.alpha)

        left_penalty = patches.Rectangle(
            (cfg.origin_x, center_y - 22),
            18, 44, fill=False,
            edgecolor=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(left_penalty)
        right_penalty = patches.Rectangle(
            (cfg.origin_x + cfg.pitch_length - 18, center_y - 22),
            18, 44, fill=False,
            edgecolor=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(right_penalty)

        left_six = patches.Rectangle(
            (cfg.origin_x, center_y - 10),
            6, 20, fill=False,
            edgecolor=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(left_six)
        right_six = patches.Rectangle(
            (cfg.origin_x + cfg.pitch_length - 6, center_y - 10),
            6, 20, fill=False,
            edgecolor=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(right_six)

        ax.plot(cfg.origin_x + 12, center_y,
                marker='o', markersize=2, color=cfg.line_color, alpha=cfg.alpha)
        ax.plot(cfg.origin_x + cfg.pitch_length - 12, center_y,
                marker='o', markersize=2, color=cfg.line_color, alpha=cfg.alpha)

        left_arc = patches.Arc(
            (cfg.origin_x + 12, center_y),
            20, 20, angle=0, theta1=-53, theta2=53,
            color=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(left_arc)
        right_arc = patches.Arc(
            (cfg.origin_x + cfg.pitch_length - 12, center_y),
            20, 20, angle=0, theta1=127, theta2=233,
            color=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(right_arc)

        goal_width = 8
        goal_offset = 2
        left_goal = patches.Rectangle(
            (cfg.origin_x - goal_offset, center_y - goal_width / 2),
            goal_offset, goal_width, fill=False,
            edgecolor=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(left_goal)
        right_goal = patches.Rectangle(
            (cfg.origin_x + cfg.pitch_length, center_y - goal_width / 2),
            goal_offset, goal_width, fill=False,
            edgecolor=cfg.line_color, lw=cfg.linewidth, alpha=cfg.alpha
        )
        ax.add_patch(right_goal)

        ax.set_xlim(cfg.origin_x - goal_offset - 5, cfg.origin_x + cfg.pitch_length + goal_offset + 5)
        ax.set_ylim(cfg.origin_y - 5, cfg.origin_y + cfg.pitch_width + 5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()

    def _draw_reference_points(self) -> None:
        """Draw reference points on the pitch for homography."""
        cfg = self.config
        ax = self.ax

        cx = cfg.origin_x + cfg.pitch_length / 2
        cy = cfg.origin_y + cfg.pitch_width / 2
        diamond_size = 10

        diamond_coords = [
            (cx, cy + diamond_size),
            (cx + diamond_size, cy),
            (cx, cy - diamond_size),
            (cx - diamond_size, cy)
        ]
        polygon = Polygon(
            diamond_coords,
            closed=True,
            fill=False,
            edgecolor='black',
            linewidth=1,
            zorder=6
        )
        ax.add_patch(polygon)

        ax.plot([cx - diamond_size, cx + diamond_size],
                [cy, cy], color='black', linewidth=1, zorder=7)
        ax.plot([cx, cx],
                [cy - diamond_size, cy + diamond_size],
                color='black', linewidth=1, zorder=7)

        origin_x = cfg.origin_x
        origin_y = cfg.origin_y
        pitch_length = cfg.pitch_length
        pitch_width = cfg.pitch_width

        points = [
            (origin_x, origin_y),
            (origin_x + pitch_length, origin_y),
            (origin_x, origin_y + pitch_width),
            (origin_x + pitch_length, origin_y + pitch_width),

            (origin_x + pitch_length / 2, origin_y + pitch_width / 2),

            (origin_x + 12, origin_y + pitch_width / 2),
            (origin_x + pitch_length - 12, origin_y + pitch_width / 2),

            (origin_x, origin_y + (pitch_width - 44) / 2),
            (origin_x, origin_y + pitch_width - (pitch_width - 44) / 2),
            (origin_x + 18, origin_y + (pitch_width - 44) / 2),
            (origin_x + 18, origin_y + pitch_width - (pitch_width - 44) / 2),

            (origin_x + pitch_length, origin_y + (pitch_width - 44) / 2),
            (origin_x + pitch_length, origin_y + pitch_width - (pitch_width - 44) / 2),
            (origin_x + pitch_length - 18, origin_y + (pitch_width - 44) / 2),
            (origin_x + pitch_length - 18, origin_y + pitch_width - (pitch_width - 44) / 2),

            (origin_x, origin_y + (pitch_width - 20) / 2),
            (origin_x, origin_y + pitch_width - (pitch_width - 20) / 2),
            (origin_x + 6, origin_y + (pitch_width - 20) / 2),
            (origin_x + 6, origin_y + pitch_width - (pitch_width - 20) / 2),

            (origin_x + pitch_length, origin_y + (pitch_width - 20) / 2),
            (origin_x + pitch_length, origin_y + pitch_width - (pitch_width - 20) / 2),
            (origin_x + pitch_length - 6, origin_y + (pitch_width - 20) / 2),
            (origin_x + pitch_length - 6, origin_y + pitch_width - (pitch_width - 20) / 2),

            (origin_x + pitch_length / 2, origin_y),
            (origin_x + pitch_length / 2, origin_y + pitch_width),
            (origin_x + pitch_length / 2, origin_y + pitch_width / 2 - 10),
            (origin_x + pitch_length / 2, origin_y + pitch_width / 2 + 10),

            (origin_x + 18, origin_y + 32),
            (origin_x + 18, origin_y + 48),
            (origin_x + pitch_length - 18, origin_y + 32),
            (origin_x + pitch_length - 18, origin_y + 48),

            (origin_x + 50, origin_y + 40),
            (origin_x + 70, origin_y + 40),
        ]

        vibrant_blue = "#0000FF"
        vibrant_pink = "#FF1493"
        vibrant_yellow = "#FFFF00"
        point_size = 100
        edge_color = "black"

        for (x, y) in points:
            if x < origin_x + pitch_length / 3:
                color = vibrant_yellow
            elif x > origin_x + 2 * pitch_length / 3:
                color = vibrant_blue
            else:
                color = vibrant_pink

            ax.scatter(
                x, y,
                s=point_size,
                color=color,
                edgecolors=edge_color,
                linewidth=1.5,
                zorder=8
            )


if __name__ == "__main__":
    drawer = HomographyPitchDrawer()
    drawer.draw_pitch()

    pitch_np = drawer.get_numpy_representation()
    # pitch_np shape: (height, width, 3)

    # example_image = drawer.load_image("reports/screenshot_2025-03-18.png")
    # drawer.save_figure("homography_pitch.png")
    drawer.show_figure()
