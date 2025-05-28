# src/process/train_dataset_loader.py

import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import DataConfig
from src.model.perspect.train_pt import PointsDataset   # loads img_paths for you :contentReference[oaicite:0]{index=0}
from src.model.batch import Sample, BatchProcessor  # reuses your Sample + collate_fn

class TrainDatasetLoader(Dataset):
    """
    Streams your PointsDataset images through the real-time inference pipeline.
    Arguments:
      - config:      a DataConfig (to grab width, height, batch_size, shuffle)
      - root:        path to the roboflow folder (same as cfg.dataset_folder)
      - split:       "train" or "valid"
      - repeats:     how many times to repeat each image in a row
    """
    def __init__(self,
                 config: DataConfig,
                 root:    str,
                 split:   str = "train",
                 repeats: int = 4):
        self.config  = config
        self.ds      = PointsDataset(root, split)
        self.batch_size = config.batch_size
        self.shuffle    = config.shuffle

        # build exactly the same transform you had in DatasetLoader
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.height, config.width)),
            transforms.ToTensor(),
        ])

        # make an index list that repeats each example `repeats`×
        self.indices = [
            i
            for i in range(len(self.ds))
            for _ in range(repeats)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Sample:
        actual = self.indices[idx]
        img_p  = self.ds.img_paths[actual]

        # read + convert → exactly as DatasetLoader did
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        img_t = self.transform(img)

        # use the same Sample dataclass
        return Sample(
            frame_id = actual,
            timestamp= float(actual),
            image    = img_t
        )

    def load(self) -> DataLoader:
        """
        Returns a DataLoader you can plug straight into RealTimeInference.
        It uses your BatchProcessor.collate_fn so `batch.frame_id` works.
        """
        return DataLoader(
            dataset     = self,
            batch_size  = self.batch_size,
            shuffle     = self.shuffle,
            num_workers = 4,
            collate_fn  = BatchProcessor.collate_fn
        )
