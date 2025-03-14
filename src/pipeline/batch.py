import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

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
            video_path: str, 
            image_size: Tuple[int, int] = (640, 352),
            skip_sec: int = 0,
            target_fps: int = 10,
            ):
        
        self.logger = setup_logger("dataset_loader.log")
        self.video_path = video_path
        self.image_size = image_size
        self.skip_sec = skip_sec
        self.target_fps = target_fps

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
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

    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.logger = setup_logger("batch_processor.log")
        self.model = model.to(device)
        self.device = device

    def process_batch(self, batch: Sample) -> List[Dict[str, torch.Tensor]]:
        """Runs inference on a batch and returns detections."""
        self.model.eval()

        with torch.no_grad():
            images = batch.image.to(self.device)
            outputs = self.model(images)

        return outputs


def collate_fn(batch: List[Sample]) -> Sample:
    """Custom collate function to batch Sample objects."""
    images = torch.stack([sample.image for sample in batch])
    frame_ids = torch.tensor([sample.frame_id for sample in batch])
    timestamps = torch.tensor([sample.timestamp for sample in batch])

    return Sample(frame_id=frame_ids, timestamp=timestamps, image=images)


if __name__ == "__main__":

    game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "00--raw" / "videos" / game_name

    print(f"Loading file from: {file_path}")

    dataset = DatasetLoader(video_path=file_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"Dataset contains {len(dataset)} frames.")

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1} shape: {batch['images'].shape}")
        if i == 4:
            break