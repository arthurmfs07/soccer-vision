
import ast
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
from typing import List

from src.struct.detection import Detection
from src.utils import parse_timestamp, get_actual_yolo

@dataclass
class BatchData:
    image:      torch.Tensor  # [B, C, H, W]
    label:      torch.Tensor  # [B, N, 2]
    match_ids:  torch.Tensor  # [B, ]
    periods:    torch.Tensor  # [B, ]
    timestamps: torch.Tensor  # [B, ]
    homography: torch.Tensor  # [B, 3, 3]


class Batch:

    """BatchData generator for training pipeline."""

    def __init__(self, data_dir: Path, csv_path: Path, batch_size: int = 16, image_size=(416, 416)):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.df = pd.read_csv(csv_path)
        self.df["timestamp_sec"] = self.df["timestamp"].apply(parse_timestamp)

        self.pointer = 0

    def __iter__(self):
        self.pointer = 0
        return self
    
    def __next__(self) -> BatchData:
        if self.pointer >= len(self.df):
            raise StopIteration
        
        end = min(self.pointer + self.batch_size, len(self.df))
        batch_df = self.df.iloc[self.pointer:end]
        self.pointer = end

        images, positions, match_ids, periods, timestamps = [], [], [], [], []

        while not images:
            if self.pointer >= len(self.df):
                raise StopIteration
            end = min(self.pointer + self.batch_size, len(self.df))
            batch_df = self.df.iloc[self.pointer:end]
            self.pointer = end

            for _, row in batch_df.iterrows():
                match_id = int(row["match_id"])
                period = int(row["period"])
                timestamp = float(row["timestamp_sec"])

                try:
                    image = self._load_image(match_id, period, timestamp)
                    player_pos = self._parse_freeze_frame(row["freeze_frame"])

                    images.append(image)
                    positions.append(player_pos)
                    match_ids.append(match_id)
                    periods.append(period)
                    timestamps.append(timestamp)

                except Exception as e:
                    print(f"Skipping sample {match_id} {period} {timestamp:.3f}: {e}")
                    continue

        
        max_players = max(p.shape[0] for p in positions)
        padded_positions = torch.full((len(positions), max_players, 2), -1.0)
        for i, p in enumerate(positions):
            padded_positions[i, :p.shape[0]] = p
        
        return BatchData(
            image=torch.stack(images),
            label=padded_positions,
            match_ids=torch.tensor(match_ids),
            timestamps=torch.tensor(timestamps),
            periods=torch.tensor(periods),
            homography=torch.eye(3).repeat(len(images), 1, 1), # placeholder
        )
    
    def _load_image(self, match_id: int, game_time: int, timestamp: float) -> torch.Tensor:
        """
        game_time \in {1, 2}
        timestamp: seconds from time init
        """
        match_dir = self.data_dir / "frames" / f"match_{match_id}"
        filename = f"t{game_time}_{timestamp:.3f}.jpg"
        img_path = match_dir / filename
        if not img_path.exists():
            raise FileNotFoundError(f"Missing frame: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

        
    def _parse_freeze_frame(self, text: str) -> torch.Tensor:
        try:
            parsed = ast.literal_eval(text)
            return torch.tensor(parsed, dtype=torch.float32)  # shape [N, 2]
        except Exception as e:
            raise ValueError(f"Failed to parse freeze_frame: {e}")




if __name__ == "__main__":
    data_dir = Path("data/frames")
    csv_path = Path("data/00--raw/df_on_360.csv")

    loader = Batch(data_dir, csv_path, batch_size=16)
    for batch in loader:
        print(batch.image.shape)
        print(batch.label.shape)
        break  # test with just the first mini-batch
