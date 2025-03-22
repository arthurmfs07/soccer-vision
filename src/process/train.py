from pathlib import Path
from src.logger import setup_logger
from src.model.detect.finetune import YOLOTrainer

from typing import Any, Dict, List, Tuple

class Trainer:

    def __init__(self, model, optimizer, criterion, device, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger or setup_logger("trainer")

    def finetune_yolo(self):
        data_path = Path(__file__).resolve().parents[3] / "data"
        dataset_yaml = data_path / "00--raw" / "football-players-detection.v12i.yolov8" / "data.yaml"
        trainer = YOLOTrainer(dataset_yaml, model_size="yolov8m", epochs=200, batch_size=16)
        save_path = data_path / "10--models" / "yolo_finetune2.pt"

        trainer.train()
        trainer.evaluate()
        trainer.save_model(save_path=save_path)

    def train_projection(self):
        pass