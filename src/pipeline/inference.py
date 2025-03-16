# real time inference handling
import time
import queue
import threading
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.model.detect.objdetect import ObjectDetector
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
            images = batch.image.to(self.detector.device)#  / 255  # Normalize
            
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
            batch_size: int = 4,
            target_fps: int = 10
            ):
        self.buffer = buffer
        self.visualizer = visualizer
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.running = True

    def process_frames(self):
        """Continuously fetches frames and renders at 1 sec per sec."""
        while self.running:
            start_time = time.time()

            if self.buffer.qsize() < self.max_buffer_size // 2:
                print(f"⏳ Buffer low ({self.buffer.qsize()}/{self.max_buffer_size}), waiting for more frames...")
                time.sleep(2)
                continue
            try:
                video_frame = self.buffer.get(timeout=0.1)
                self.visualizer.update(video_frame)
                self.visualizer.show()
            except queue.Empty:
                print("⏳ Buffer is empty, waiting for frames...")
                time.sleep(0.1)
                continue

            # sleep_time = max ( 0, 1/target_fps - t_process )
            elapsed_time = time.time() - start_time
            remaining_time = self.frame_interval - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

        print("✅ Visualization Process Finished")

    def stop(self):
        """Stops the visualization process."""
        self.running = False

class RealTimeInference:

    """Real-time object detection and visualization pipeline."""

    def __init__(self):
        
        self.batch_size = 4       # Size of a single inference batch
        self.max_buffer_size = 40 # Visualization buffer size (in batches)
        self.setup()


    def setup(self):
        self.data_path = Path(__file__).resolve().parents[2] / "data"

        self.yolo_path = self.data_path/ "10--models" / "yolov8_finetuned.pt"

        game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
        self.example_video_path = self.data_path / "00--raw" / "videos" / game_name
        

    def run(self):
        # Load YOLO detector
        detector = ObjectDetector(self.yolo_path, conf=0.5)

        # Load dataset
        dataset = DatasetLoader(video_path=self.example_video_path, skip_sec=100*60)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        # Shared buffer (FIFO queue)
        buffer = queue.Queue(maxsize=50)  # Adjust buffer size if needed

        # Initialize processes
        batch_inference = InferenceProcess(detector, dataloader, buffer, batch_size=self.batch_size)
        visualizer = Visualizer(PitchConfig(scale=5, linewidth=1), np.zeros((720, 1280, 3), dtype=np.uint8), detector.class_names)
        visualization = VisualizationProcess(buffer, visualizer, max_buffer_size=self.max_buffer_size, batch_size=self.batch_size)

        # Start threads
        inference_thread = threading.Thread(target=batch_inference.process_batches)
        visualization_thread = threading.Thread(target=visualization.process_frames)

        inference_thread.start()
        visualization_thread.start()

        try:
            inference_thread.join()
            visualization_thread.join()
        except KeyboardInterrupt:
            batch_inference.stop()
            visualization.stop()

if __name__ == "__main__":

    pipeline = RealTimeInference()
    pipeline.run()