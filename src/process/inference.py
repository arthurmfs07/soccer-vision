import queue
import torch
import numpy as np
import warnings
from scipy.linalg import logm, expm, LinAlgError
from typing import Literal, Optional
from torch.utils.data import DataLoader

from typing import List, Tuple
from src.visual.video import VideoFrame
from src.process.process import Process
from src.config import RealTimeConfig
from src.model.perspect.model import PerspectModel

from src.struct.shared_data import SharedAnnotations
from src.struct.utils import create_base_square

class InferenceProcess(Process):
    """Handles batch-wise inference and stores results in a buffer."""

    def __init__(
            self, 
            detector, 
            dataloader: DataLoader, 
            buffer: queue.Queue, 
            config: RealTimeConfig = RealTimeConfig(),
            shared_data: SharedAnnotations = SharedAnnotations(),
            model: Optional[PerspectModel] = None
            ):
        self.detector = detector
        self.dataloader = dataloader
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.shared_data = shared_data
        self.model = model

        self.running = True
        self.prev_log_Hs = None  # will hold list of previous log‐mats
        self.smooth_alpha: float = 0.3 # [0..1], greater = more smooth

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
            
            frame_ids   = batch.frame_id  # Extract frame IDs
            timestamps  = batch.timestamp
            images      = batch.image.to(self.config.device)#  / 255  # Normalize
            detections  = self.detector.detect(images)
            Hs = [None] * len(frame_ids)

            if self.model is not None:
                with torch.no_grad():
                    Hs_t = self.model.predict_homography(images)
                Hs_np = Hs_t.cpu().numpy().astype(np.float32)  # [B,3,3]
                Hs    = self.smooth_homographies(Hs_np)
            else:
                Hs = [None] * len(frame_ids)

            images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255
                         ).astype(np.uint8)

            for i in range(len(frame_ids)):
                vf = VideoFrame(
                    frame_id=frame_ids[i].item(),
                    timestamp=timestamps[i].item(),
                    image=images_np[i],
                    detections=[detections[i]],
                    H=Hs[i]
                )
                self.buffer.put(vf)        
        print("✅ Batch Inference Process Finished")
        self.phase = "done"


    def smooth_homographies(self, Hs_np: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Lie‐group exponential smoothing with error handling.
        Hs_np: numpy array [B,3,3] (or list of B arrays)
        """
        smoothed, new_prev = [], []

        # first batch => just initialize
        if self.prev_log_Hs is None:
            self.prev_log_Hs = [None] * len(Hs_np)

        for L_prev, H in zip(self.prev_log_Hs, Hs_np):
            if H is None:
                smoothed.append(None)
                new_prev.append(L_prev)
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    L = logm(H)
            except LinAlgError as e:
                smoothed.append(H)
                new_prev.append(L_prev)
                continue

            L_s = L if L_prev is None else (
                (1.0 - self.smooth_alpha) * L + self.smooth_alpha * L_prev
            )
            H_s = expm(L_s).real
            H_s = (H_s / H_s[2, 2]).astype(np.float32)
            smoothed.append(H_s)
            new_prev.append(L_s)
        
        self.prev_log_Hs = new_prev
        return smoothed


    def stop(self):
        """Stops the batch inference loop."""
        self.running = False



if __name__ == "__main__":

    pipeline = InferenceProcess()
    pipeline.run()