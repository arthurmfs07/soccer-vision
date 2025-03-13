import torch

from src.pipeline.batch import DataLoader
from src.logger import setup_logger


class Trainer:
    """Handles the training pipeline for object detection."""
    

    # REBUILD THIS

    def __init__(self, 
                 model: torch.nn.Module, 
                 dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: torch.nn.Module, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.logger = setup_logger("trainer.log")
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self) -> float:
        """Runs one training epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.dataloader:
            images, labels = batch.image.to(self.device), batch.labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.dataloader)
        self.logger.info(f"Epoch completed. Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, epochs: int):
        """Trains the model for multiple epochs."""
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{epochs}")
            avg_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch+1} finished. Loss: {avg_loss:.4f}")
