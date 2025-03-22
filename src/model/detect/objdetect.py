# YOLO model: loading, training, inference

import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from src.logger import setup_logger


@dataclass
class Detection:
    """Stores detection results for a single frame."""
    boxes: np.ndarray  # Boundig boxes (x1, y1, x2, y2)
    confidences: np.ndarray # Confidence scores
    classes: np.ndarray # Class IDs




class ObjectDetector:
    MODEL_PATH = Path(__file__).resolve().parent / "data" / "03--models" / "yolov8.pt"
    YOLO_WEIGHTS_URL = "yolov8m.pt"

    class_names = {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee"
    }

    def __init__(
            self,
            model_path: Path = Path("yolov8m.pt"),
            conf: float = 0.01,
            device: str = "cuda"
        ):
        """Loads YOLOv8 model for object detection."""
        self.logger = setup_logger("api.log")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf = conf

        self.model_path = model_path or self.MODEL_PATH
        self._ensure_model_downloaded()

        self.model = YOLO(model_path).to(self.device)
        self.logger.info(f"Loaded YOLO model : {model_path.name}")

    def detect(self, images: torch.Tensor) -> List[Detection]:
        """
        Runs object detection on a batch of images (preprocessed).
        images.shape : [B, C, H, W]
        """
        if images.device != self.device:
            images = images.to(self.device)

        results = self.model(images, verbose=False, conf=self.conf)
        detections = []

        for result in results:
            detections.append(Detection(
                boxes=result.boxes.xyxy.cpu().numpy(),       # Bounding box (x1, y1, x2, y2)
                confidences=result.boxes.conf.cpu().numpy(), # Confidence scores
                classes=result.boxes.cls.cpu().numpy()       # Class
            ))
        return detections
    
    def _ensure_model_downloaded(self):
        """Ensures YOLO model is properly downloaded and saved."""
        if not os.path.exists(self.model_path):
            self.logger.info(f"Model not found at {self.model_path}. Downloading YOLO model...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            model = YOLO(self.YOLO_WEIGHTS_URL)
            model.save(self.model_path)

            self.logger.info(f"Model downloaded and saved to {self.model_path}")


if __name__ == "__main__":
    from model.batch import DatasetLoader, collate_fn

    data_path = Path(__file__).resolve().parents[3] / "data"
    model_path = data_path / "10--models" / "yolov8_finetuned.pt"
    detector = ObjectDetector(model_path)

    game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
    file_path = data_path / "00--raw" / "videos" / game_name

    print(f"Loading file from: {file_path}")

    dataset = DatasetLoader(video_path=file_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


    for i, batch in enumerate(dataloader):
        detections = detector.detect(batch.image)
        print(detections)
        break


