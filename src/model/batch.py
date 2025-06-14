import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from src.config import RealTimeConfig
from src.model.add_coords import AddCoords

from src.logger import setup_logger


@dataclass
class Sample:
    """Encapsulate a single video frame for inference."""
    frame_id: int
    timestamp: float
    image: torch.Tensor


class DatasetLoader(Dataset):
    """Loads video frames and transforms them into PyTorch tensors."""
    
    def __init__(
            self, 
            config: RealTimeConfig = RealTimeConfig()
            ):
        
        self.logger = setup_logger("dataset_loader.log")
        self.config = config
        self.video_dir = config.video_dir
        self.video_name = config.video_name
        self.video_path = Path(self.video_dir) / self.video_name
        self.device = config.device

        self.target_fps = config.target_fps
        self.batch_size = config.batch_size
        self.width = config.imgsz
        self.height = config.imgsz
        self.skip_sec = config.skip_sec

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            # AddCoords()
        ])

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Failed to open video: {self.video_path}")

        self.original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval = max(1, self.original_fps // self.target_fps)
        
        self.start_frame = min(self.skip_sec * self.original_fps, self.frame_count)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        self.logger.info(f"📂 Video: {self.video_path} | Frames: {self.frame_count}")

    def load(self):
        """Load the model weights."""
        self.dataloader = DataLoader(
            dataset=self, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=BatchProcessor.collate_fn
            )
        return self.dataloader


    def __len__(self) -> int:
        return (self.frame_count - self.start_frame) // self.frame_interval

    def __getitem__(self, idx: int) -> Sample:
        """Loads a single frame lazily on demand."""
        frame_idx = self.start_frame + idx * self.frame_interval
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError(f"❌ Failed to read frame {frame_idx} from {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transform(frame)

        return Sample(frame_id=frame_idx, timestamp=frame_idx / self.original_fps, image=image)


    def __del__(self):
        """Ensure the video file is properly closed."""
        if hasattr(self, "cap"):
            self.cap.release()



class BatchProcessor:
    """Handles batch inference and dataset processing."""

    def __init__(self, model: torch.nn.Module, config: RealTimeConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.device = config.device
        self.logger = setup_logger("batch_processor.log")
        self.model = model.to(self.device)

    def process_batch(self, batch: Sample) -> List[Dict[str, torch.Tensor]]:
        """Runs inference on a batch and returns detections."""
        self.model.eval()

        with torch.no_grad():
            images = batch.image.to(self.device)
            outputs = self.model(images)

        return outputs
    
    @staticmethod
    def collate_fn(batch: List[Sample]) -> Sample:
        """Custom collate function to batch Sample objects."""
        images = torch.stack([sample.image for sample in batch])
        frame_ids = torch.tensor([sample.frame_id for sample in batch])
        timestamps = torch.tensor([sample.timestamp for sample in batch])

        return Sample(frame_id=frame_ids, timestamp=timestamps, image=images)


if __name__ == "__main__":

    from src.utils import load_abs_path
    project_path = load_abs_path()

    dataset = DatasetLoader(config=RealTimeConfig())
    
    print(f"Dataset contains {len(dataset)} frames.")

