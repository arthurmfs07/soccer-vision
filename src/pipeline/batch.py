import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from src.logger import setup_logger

@dataclass
class Sample:
    """Encapsulates a sample in the dataset (image and annotations)."""
    image: torch.Tensor
    labels: torch.Tensor


class DatasetLoader(Dataset):
    """Loads video frames and transforms them into PyTorch tensors."""
    
    def __init__(self, 
                 video_path: str, 
                 annotation_path: Optional[str] = None, 
                 image_size: Tuple[int, int] = (640, 640)):
        
        self.logger = setup_logger("dataset_loader.log")
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.frames = self._extract_frames()
        self.annotations = self._load_annotations() if annotation_path else None
        self.logger.info(f"Loaded dataset with {len(self.frames)} frames.")

    def _extract_frames(self) -> List[np.ndarray]:
        """Extracts frames from a video file."""
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames

    def _load_annotations(self) -> List[np.ndarray]:
        """Loads annotation data if provided."""
        return np.load(self.annotation_path, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Sample:
        """Returns a single sample with image and corresponding label (if available)."""
        image = self.transform(self.frames[idx])
        label = torch.tensor(self.annotations[idx]) if self.annotations is not None else torch.zeros(1)
        return Sample(image=image, labels=label)


class BatchProcessor:
    """Handles batch inference and dataset processing."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.logger = setup_logger("batch_processor.log")
        self.model = model.to(device)
        self.device = device

    def process_batch(self, batch: List[Sample]) -> List[Dict[str, torch.Tensor]]:
        """Runs inference on a batch and returns detections."""
        self.model.eval()
        
        with torch.no_grad():
            images = torch.stack([sample.image.to(self.device) for sample in batch])
            outputs = self.model(images)
        
        return outputs


if __name__ == "__main__":

    game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.webm"
    file_path = Path(__file__).resolve().parent / "soccer-vision" / "data" / "00--raw" / "videos" / game_name

