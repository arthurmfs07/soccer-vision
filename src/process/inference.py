import queue
import torch
import cv2
import numpy as np
import warnings
from scipy.linalg import logm, expm, LinAlgError
from typing import Literal, Optional
from torch.utils.data import DataLoader
import torch.nn.functional as F

from typing import List, Tuple
from src.visual.video import VideoFrame
from src.process.process import Process
from src.config import RealTimeConfig

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
            model: Optional["PerspectModel"] = None
            ):
        self.detector = detector
        self.dataloader = dataloader
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.shared_data = shared_data
        self.model = model
        self.model.eval()


        self.running = True
        self.prev_log_Hs = None  # will hold list of previous log‐mats

        self.smooth_alpha: float = 0.8 # [0..1], greater = more smooth
        self.conf_th:      float = 0.5

        self.model.conf_th = self.conf_th

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
            images      = batch.image.to(self.config.device)
            detections  = self.detector.detect(images[:,:3])
            Hs = [None] * len(frame_ids)

            if self.model is not None:
                with torch.no_grad():
                    Hs_t, coords_t, vis_log_t = self.model.predict(images, predict_H=True)
                Hs_np = Hs_t.cpu().numpy().astype(np.float32)  # [B,3,3]
                # Hs    = self.smooth_homographies(Hs_np)
                Hs = list(Hs_np)
                coords_np = coords_t.cpu().numpy().astype(np.float32)
                if vis_log_t is not None:
                    vis_p_t = torch.sigmoid(vis_log_t)
                    vis_mask  = (vis_p_t.cpu().numpy() > self.conf_th)
                else:
                    vis_mask = None
            else:
                Hs = [None] * len(frame_ids)
                coords_np = np.zeros((len(batch.frame_id),32,2), np.float32)
                vis_mask  = np.zeros((len(batch.frame_id),32),   bool)

            images_np = (images[:,:3]
                         .cpu()
                         .permute(0, 2, 3, 1)
                         .numpy() * 255
                         ).astype(np.uint8)
            
            feet_list = self._extract_feet(detections)

            for i in range(len(frame_ids)):
                snap = SharedAnnotations()

                # YOLO boxes (yellow)
                det = detections[i]
                snap.yolo_detections = [
                    {"bbox": tuple(map(float, bb)), "class": int(cls)}
                    for bb, cls in zip(det.boxes, det.classes)
                ]

                # Homography
                if Hs[i] is None:
                    Hs[i] = np.eye(3, dtype=np.float32)
                snap.H_video2field = Hs[i]

                # CNN points (blue)
                if vis_mask is not None: # point model
                    pts     = coords_np[i]        # (32,2)
                    mask    = vis_mask[i][:,None] # (32,1)
                    blue    = np.where(mask, pts, -1.0) # (32,2)
                    snap.numbered_video_points["blue"] = blue.astype(np.float32)

                else: # square model
                    snap.field_points["blue"] = coords_np[i]

                fld_i = self.warp_points_np(Hs[i], feet_list[i])
                snap.field_points["yellow"] = fld_i

                vf = VideoFrame(
                    frame_id    = frame_ids[i].item(),
                    timestamp   = timestamps[i].item(),
                    image       = images_np[i],
                    annotations = snap
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
    


    @staticmethod
    def warp_points_np(
        H: np.ndarray, 
        pts: np.ndarray
    ) -> np.ndarray:
        """
        Warp Nx2 normalized points by a 3x3 homography H
        """
        if pts.size == 0:
            return pts.reshape(0, 2).astype(np.float32)
        src = pts.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(src, H.astype(np.float32))
        return dst.reshape(-1, 2).astype(np.float32)


    def _extract_feet(
        self,
        dets: List
    ) -> List[np.ndarray]:
        feet = []
        for det in dets:
            pts = [((x1+x2)/2, y2) for x1,y1,x2,y2 in det.boxes]
            feet.append(np.array(pts, np.float32))
        return feet


    def stop(self):
        """Stops the batch inference loop."""
        self.running = False



if __name__ == "__main__":

    pipeline = InferenceProcess()
    pipeline.run()