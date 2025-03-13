import torch
import cv2
import numpy as np

from src.config import VideoConfig

class Video:
    """Tensor wrapper"""
    def __init__(self, filepath: str, config: VideoConfig):
        self.filepath = filepath
        self.config = config
        self.device = config.device

        self.framerate = config.fps
        self.cap = cv2.VideoCapture(filepath)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {filepath}")

        self.original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_interval = max(1, self.original_fps // self.framerate)

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

    def _process_frame(self, frame: np.ndarray):
        """Convert OpenCV frame to PyTorch tensor [C, H, W]"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame, device=self.device).float() / 255.0
        frame = frame.permute(2, 0, 1) # Convert [H, W, C] -> [C, H, W]
        return frame
    
    def load(self):
        """Load entire video as a tensor in [T, C, H, W] format"""
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.current_frame_idx % self.frame_interval == 0:
                frames.append(self._process_frame(frame))
            self.current_frame_idx += 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video position
        self.current_frame_idx = 0
        self.video = torch.stack(frames)  # [T, C, H, W]
        return self.video

    def next(self, batch_size:int = 1):
        """Load the next batch of frames"""
        frames = []
        for _ in range(batch_size):
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.current_frame_idx % self.frame_interval == 0:
                frames.append(self._process_frame(frame))
            self.current_frame_idx += 1

        if not frames:
            return None  # End of video

        return torch.stack(frames)  # [B, C, H, W] format

    def reset(self):
        """ Reset video to the beginning """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0

    def close(self):
        """ Release video file """
        self.cap.release()