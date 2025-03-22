
from typing import Tuple

def get_color(color_name: str) -> Tuple[int, int, int]:
    """Convert color name to BGR tuple."""
    colors = {
        "red":        (0, 0, 255),
        "green":      (0, 128, 0),
        "lightgreen": (144, 238, 144),
        "blue":       (255, 0, 0),     
        "lightblue":  (173, 216, 230),
        "yellow":     (0, 255, 255),
        "orange":     (0, 165, 255),   
        "purple":     (128, 0, 128),   
        "black":      (0, 0, 0),       
        "white":      (255, 255, 255), 
        "offwhite":   (210, 210, 210), 
        "gray":       (128, 128, 128),
        "darkgray":   (50, 50, 50),    
    }
    return colors.get(color_name.lower(), (0, 0, 0))  # Default to black
