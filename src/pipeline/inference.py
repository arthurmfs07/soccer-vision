# real time inference handling
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.model.objdetect import ObjectDetector
from src.pipeline.batch import DatasetLoader, collate_fn
from src.visualizer.visualizer import Visualizer
from src.visualizer.field import PitchConfig

def main():
    """Runs object detection and visualizes results in real-time."""
    model_path = Path(__file__).resolve().parent / "data" / "03--models" / "yolov8.pt"
    game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "00--raw" / "videos" / game_name

    print(f"🎥 Loading video from: {file_path}")

    detector = ObjectDetector(model_path)
    dataset = DatasetLoader(video_path=file_path, skip_sec=100*60)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    config = PitchConfig(scale=5, linewidth=1)

    first_frame = next(iter(dataloader))["images"][0].permute(1, 2, 0).numpy() * 255  # Convert to OpenCV format
    visualizer = Visualizer(PitchConfig(scale=5, linewidth=1), first_frame.astype(np.uint8))

    def frame_generator():
        """Yields video frames with detections applied."""
        for batch in dataloader:
            frame = batch["images"][0].permute(1, 2, 0).numpy()  # Convert (C, H, W) → (H, W, C)
            frame = (frame * 255).clip(0, 255).astype(np.uint8)  # Normalize and convert to uint8
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            images_for_detection = batch["images"].to(detector.device) / 255.0  # Normalize to [0,1]
            detections = detector.detect(images_for_detection)
            visualizer.video_visualizer.annotate(detections)
            yield frame.astype(np.uint8)

    visualizer.show(frame_generator())

if __name__ == "__main__":
    main()
