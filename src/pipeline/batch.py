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
    
    def __init__(
            self, 
            video_path: str, 
            annotation_path: Optional[str] = None, 
            image_size: Tuple[int, int] = (640, 640),
            skip_sec: int = 0,
            ):
        
        self.logger = setup_logger("dataset_loader.log")
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.image_size = image_size
        self.skip_sec = skip_sec

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Failed to open video: {self.video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_frame = min(self.skip_sec * self.fps, self.frame_count)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        self.annotations = self._load_annotations() if annotation_path else None

        self.logger.info(f"📂 Video: {self.video_path} | Frames: {self.frame_count}")

    def _load_annotations(self) -> List[np.ndarray]:
        """Loads annotation data if provided."""
        return np.load(self.annotation_path, allow_pickle=True)

    def __len__(self) -> int:
        return self.frame_count - self.start_frame

    def __getitem__(self, idx: int) -> Sample:
        """Loads a single frame lazily on demand."""
        frame_idx = self.start_frame + idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError(f"❌ Failed to read frame {frame_idx} from {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transform(frame)
        label = torch.tensor(self.annotations[idx]) if self.annotations is not None else torch.zeros(1)

        return Sample(image=image, labels=label)

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

    def process_batch(self, batch: List[Sample]) -> List[Dict[str, torch.Tensor]]:
        """Runs inference on a batch and returns detections."""
        self.model.eval()

        with torch.no_grad():
            images = torch.stack([sample.image.to(self.device) for sample in batch])
            outputs = self.model(images)

        return outputs


def collate_fn(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    """Custom collate function to batch Sample objects."""
    images = torch.stack([sample.image for sample in batch])
    labels = torch.stack([sample.labels for sample in batch])
    return {"images": images, "labels": labels}


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