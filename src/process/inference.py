import queue
import cv2
import torch
import numpy as np
from scipy.linalg import logm, expm
from typing import Literal, Optional
from torch.utils.data import DataLoader

from typing import List, Tuple
from src.visual.video import VideoFrame
from src.process.process import Process
from src.config import RealTimeConfig
from src.model.perspect.cnn import CNN

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
            shared_data: SharedAnnotations = SharedAnnotations(),
            cnn_model: Optional[CNN] = None
            ):
        self.detector = detector
        self.dataloader = dataloader
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.shared_data = shared_data
        self.cnn_model = cnn_model

        self.running = True
        self.prev_log_Hs = None  # will hold list of previous log‐mats
        self.smooth_alpha: float = 0.2

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
            images = batch.image.to(self.config.device)#  / 255  # Normalize
            detections_batch = self.detector.detect(images)

            Hs = [None] * len(frame_ids)
            if self.cnn_model is not None:
                with torch.no_grad():
                    inp = batch.image.to(self.config.device, dtype=torch.float32)
                    pred_norm = self.cnn_model(inp)       # [B,4,2]

                    from src.visual.field import PitchConfig
                    from src.struct.transform import TransformUtils
                    tpl = PitchConfig()
                    pred_m = (
                        pred_norm.cpu().numpy()
                        * np.array([tpl.length, tpl.width], dtype=np.float32)
                    )

                    h_px, w_px = images.shape[2], images.shape[3]
                    base_sq_px = TransformUtils.get_base_square_px((h_px, w_px))

                    for i in range(len(frame_ids)):
                        H, _ = cv2.findHomography(
                            base_sq_px.astype(np.float32),
                            pred_m[i].astype(np.float32),
                            cv2.RANSAC
                        )
                        Hs[i] = H

                Hs = self.smooth_homographies(Hs)


            images_np = (batch.image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # Store results in buffer
            for i in range(len(frame_ids)):
                frame_instance = VideoFrame(
                    frame_id=frame_ids[i].item(),
                    timestamp=timestamps[i].item(),
                    image=images_np[i],
                    detections=[detections_batch[i]],
                    H=Hs[i]
                )
                self.buffer.put(frame_instance)        
        print("✅ Batch Inference Process Finished")



    def smooth_homographies(self, Hs):
        """
        Lie‐group smoothing of a batch of homographies.
        Hs: list of (3×3) numpy arrays or None
        Returns: list of smoothed 3×3 homographies (same length)
        """
        smoothed = []

        # first batch => just initialize
        if self.prev_log_Hs is None:
            self.prev_log_Hs = []
            for H in Hs:
                if H is None:
                    self.prev_log_Hs.append(None)
                    smoothed.append(None)
                else:
                    L = logm(H)              # matrix‐log
                    self.prev_log_Hs.append(L)
                    smoothed.append(H)
            return smoothed

        # subsequent batches => blend logs
        new_prev = []
        for L_prev, H in zip(self.prev_log_Hs, Hs):
            if H is None:
                # no new estimate, carry forward
                new_prev.append(L_prev)
                smoothed.append(None)
                continue

            L = logm(H)
            if L_prev is None:
                L_s = L
            else:
                # exponential smoothing in Lie‐algebra
                L_s = self.smooth_alpha * L + (1 - self.smooth_alpha) * L_prev

            H_s = expm(L_s)           # back to projective group
            H_s = H_s / H_s[2,2]      # renormalize scale
            new_prev.append(L_s)
            smoothed.append(H_s)

        self.prev_log_Hs = new_prev
        return smoothed



    def stop(self):
        """Stops the batch inference loop."""
        self.running = False



if __name__ == "__main__":

    pipeline = InferenceProcess()
    pipeline.run()