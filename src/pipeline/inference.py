# real time inference handling
import cv2
import time
import queue
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from src.model.objdetect import ObjectDetector
from src.visualizer.visualizer import Visualizer
from src.visualizer.video import VideoFrame
from src.pipeline.batch import DatasetLoader, collate_fn
from src.visualizer.field import PitchConfig



class InferenceProcess:
    """Handles batch-wise inference and stores results in a buffer."""

    def __init__(
            self, 
            detector, 
            dataloader: DataLoader, 
            buffer: queue.Queue, 
            batch_size: int = 4
            ):
        self.detector = detector
        self.dataloader = dataloader
        self.buffer = buffer
        self.batch_size = batch_size
        self.running = True


    def process_batches(self):
        """Loads frames in batches, performs inference, and stores results."""
        print("🚀 Inference Process Started")
        for batch in self.dataloader:
            if not self.running:
                break
            
            frame_ids = batch.frame_id  # Extract frame IDs
            timestamps = batch.timestamp
            images = batch.image.to(self.detector.device) / 255  # Normalize
            
            # Perform detection

            detections_batch = self.detector.detect(images)

            images_np = (batch.image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # Store results in buffer
            for i in range(len(frame_ids)):
                frame_instance = VideoFrame(
                    frame_id=frame_ids[i].item(),
                    timestamp=timestamps[i].item(),
                    image=images_np[i],
                    detections=detections_batch[i]
                )
                self.buffer.put(frame_instance)
        
        print("✅ Batch Inference Process Finished")

    def stop(self):
        """Stops the batch inference loop."""
        self.running = False


class VisualizationProcess:
    """Retrieves results from the buffer and visualizes in real-time."""
    
    def __init__(
            self, 
            buffer: queue.Queue, 
            visualizer, 
            max_buffer_size: int = 40,
            batch_size: int = 4):
        self.buffer = buffer
        self.visualizer = visualizer
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.running = True

    def process_frames(self):
        """Continuously fetches frames and renders at 1 sec per sec."""
        while self.running:
            try:
                video_frame = self.buffer.get(timeout=0.1)
                
                self.visualizer.update(video_frame)
                self.visualizer.show()
            except queue.Empty:
                print("⏳ Buffer is empty, waiting for frames...")
                time.sleep(0.01)
            
            start_time = time.time()

            for _ in range(min(self.batch_size, self.buffer.qsize())):
                try:
                    video_frame = self.buffer.get_nowait()
                    self.visualizer.update(video_frame)
                    self.visualizer.show()

                except queue.Empty:
                    break # No more frames to process

            elapsed_time = time.time() - start_time
            target_fps = 30
            sleep_time = max(0, (1 / target_fps) - elapsed_time)
            time.sleep(sleep_time)
        
        print("✅ Visualization Process Finished")

    def stop(self):
        """Stops the visualization process."""
        self.running = False


def main():
    # Paths
    model_path = Path(__file__).resolve().parent / "data" / "03--models" / "yolov8.pt"
    game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
    video_path = Path(__file__).resolve().parent.parent.parent / "data" / "00--raw" / "videos" / game_name

    batch_size = 4
    max_buffer_size = 40

    # Load YOLO detector
    detector = ObjectDetector(model_path)

    # Load dataset
    dataset = DatasetLoader(video_path=video_path, skip_sec=100*60)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Shared buffer (FIFO queue)
    buffer = queue.Queue(maxsize=50)  # Adjust buffer size if needed

    # Initialize processes
    batch_inference = InferenceProcess(detector, dataloader, buffer, batch_size=batch_size)
    visualizer = Visualizer(PitchConfig(scale=5, linewidth=1), np.zeros((720, 1280, 3), dtype=np.uint8))
    visualization = VisualizationProcess(buffer, visualizer, max_buffer_size=max_buffer_size, batch_size=batch_size)

    # Start threads
    inference_thread = threading.Thread(target=batch_inference.process_batches)
    visualization_thread = threading.Thread(target=visualization.process_frames)

    inference_thread.start()
    visualization_thread.start()

    # Wait for threads to finish
    try:
        inference_thread.join()
        visualization_thread.join()
    except KeyboardInterrupt:
        batch_inference.stop()
        visualization.stop()

if __name__ == "__main__":
    main()
