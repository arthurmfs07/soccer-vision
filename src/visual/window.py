from abc import ABC, abstractmethod

class Window(ABC):

    """Handle the visualization of a Frame."""


    def resize_to_width(self, new_width: int):
        self.frame.update_currents()
        current_width = self.frame.current_width
        if current_width <= 0:
            return
        
        scale_factor = float(new_width) / float(current_width)
        old_scale = self.frame._scale
        self.frame.update_scale(old_scale * scale_factor)