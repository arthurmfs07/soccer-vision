# YOLO model: loading, training, inference

import os
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
from src.logger import setup_logger

class ObjectDetector:
    MODEL_PATH = Path(__file__).resolve().parent / "data" / "03--models" / "yolov8.pt"
    YOLO_WEIGHTS_URL = "yolov8n.pt"
    
    def __init__(
            self,
            model_path: Path = Path("yolov8s.pt"),
            device: str = "cuda"
        ):
        """Loads YOLOv8 model for object detection."""
        self.logger = setup_logger("api.log")
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model_path = model_path or self.MODEL_PATH
        self._ensure_model_downloaded()

        self.model = YOLO(model_path).to(self.device)
        self.logger.info(f"Loaded YOLO model : {model_path.name}")

    def detect(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Runs object detection on a batch of images (preprocessed).
        images.shape : [B, C, H, W]
        """
        if images.device != self.device:
            images = images.to(self.device)

        results = self.model(images)
        detections = []

        for result in results:
            detections.append({
                "boxes": result.boxes.xyxy.cpu().numpy(),       # Bounding box (x1, y1, x2, y2)
                "confidences": result.boxes.conf.cpu().numpy(), # Confidence scores
                "classes": result.boxes.cls.cpu().numpy()       # Class
            })
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
    from src.pipeline.batch import DatasetLoader

    model_path = Path(__file__).resolve().parent / "data" / "03--models" / "yolov8.pt"
    detector = ObjectDetector(model_path)

    game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.webm"
    file_path = Path(__file__).resolve().parent / "soccer-vision" / "data" / "00--raw" / "videos" / game_name

    dataset = DatasetLoader(file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images in dataloader:
        detections = detector.detect(images)
        print(detections)
        break
