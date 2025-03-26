import queue
import numpy as np
from typing import Literal
from torch.utils.data import DataLoader

from typing import List, Tuple
from src.visual.video import VideoFrame
from src.process.process import Process
from src.config import RealTimeConfig

from src.struct.shared_data import SharedAnnotations
from src.struct.utils import annotate_frame_with_detections

class InferenceProcess(Process):
    """Handles batch-wise inference and stores results in a buffer."""

    def __init__(
            self, 
            detector, 
            dataloader: DataLoader, 
            buffer: queue.Queue, 
            config: RealTimeConfig = RealTimeConfig(),
            shared_data: SharedAnnotations = SharedAnnotations()
            ):
        self.detector = detector
        self.dataloader = dataloader
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.shared_data = shared_data
        self.running = True

        self.video_detection_pts: List[Tuple[int, int]]

        self.phase: Literal["inference", "annotation", "done"] = "inference"

    def on_mouse_click(self, x: int, y: int) -> None:
        """Handles mouse click events in the visualizer."""
        pass

    def is_done(self) -> bool:
        """Checks whether the process is completed."""
        return self.phase == "done"
        
        
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
                    detections=[detections_batch[i]]
                )
                self.buffer.put(frame_instance)        
        print("✅ Batch Inference Process Finished")


    def stop(self):
        """Stops the batch inference loop."""
        self.running = False



if __name__ == "__main__":

    pipeline = InferenceProcess()
    pipeline.run()