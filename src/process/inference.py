# src/process/inference.py
from __future__ import annotations
import queue, warnings
from typing import List, Optional, Tuple, Literal, Union

import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.linalg import logm, expm, LinAlgError

from src.visual.video import VideoFrame
from src.process.process import Process
from src.config import RealTimeConfig, InferenceConfig, VisualizationConfig
from src.struct.shared_data import SharedAnnotations
from src.model.team_cluster import TeamClusterer


class InferenceProcess(Process):
    """Batch‐wise detection, homography, and annotation pipeline."""

    def __init__(
        self,
        detector,
        dataloader: DataLoader,
        buffer: queue.Queue,
        shared_data: SharedAnnotations = SharedAnnotations(),
        model: Optional[Union["PerspectModel", "YOLOModel"]] = None,
    ):
        self.detector = detector
        self.dataloader = dataloader
        self.buffer = buffer
        self.rt_cfg = RealTimeConfig()
        self.inf_cfg = InferenceConfig()
        self.shared_data = shared_data
        self.model = model
        self.model.eval()

        self.conf_th = self.inf_cfg.kp_conf_th
        self.model.conf_th = self.conf_th

        self.player_class_id: int = 2
        self.tc_conf_th = self.inf_cfg.tc_conf_th
        self.team_clusterer = TeamClusterer(self.tc_conf_th)

        self.vis_cfg = VisualizationConfig()

        self.prev_log_Hs: Optional[List[np.ndarray]] = None

        self.running = True
        self.phase: Literal["inference", "annotation", "done"] = "inference"

    # event hooks kept
    def on_mouse_click(self, x: int, y: int) -> None: ...
    def is_done(self) -> bool: return self.phase == "done"

    # ──────────────────────────────────────────────────────────────
    def process_batches(self) -> None:
        for batch in self.dataloader:
            if not self.running:
                break

            frame_ids, timestamps = batch.frame_id, batch.timestamp
            images = batch.image.to(self.rt_cfg.device)          # B×3×H×W
            detections = self.detector.detect(images[:, :3])

            # homography model
            if self.model:
                with torch.no_grad():
                    Hs_t, coords_t, vis_p_t = self.model.predict(images, True)
                Hs = list(Hs_t.cpu().numpy().astype(np.float32))
                coords_np = coords_t.cpu().numpy().astype(np.float32)
                vis_mask = (
                    vis_p_t.cpu().numpy() > self.conf_th if vis_p_t is not None else None
                )
            else:
                Hs = [None] * len(frame_ids)
                coords_np = np.zeros((len(frame_ids), 32, 2), np.float32)
                vis_mask = None

            imgs = (images[:, :3].cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                np.uint8
            )

            # feet for all detections
            feet_all = self._extract_feet(detections)

            # boolean masks for player class
            ply_mask = [
                det.classes.astype(np.int32) == self.player_class_id for det in detections
            ]
            # players only
            ply_boxes = [det.boxes[m] for det, m in zip(detections, ply_mask)]
            ply_feet = [f[m] for f, m in zip(feet_all, ply_mask)]

            # team clustering
            batch_masks = self.team_clusterer.cluster_batch(imgs, ply_boxes)

            for i, (m0, m1) in enumerate(batch_masks):
                snap = SharedAnnotations()

                # show all detections
                det = detections[i]
                if self.vis_cfg.show_detections:
                    snap.yolo_detections = [
                        {"bbox": tuple(map(float, b)), "class": int(c)}
                        for b, c in zip(det.boxes, det.classes)
                    ]

                # blue numbered field points
                if self.vis_cfg.show_field_keypoints:
                    from src.visual.field import FieldVisualizer, PitchConfig
                    canon = FieldVisualizer(PitchConfig())._reference_model_pts()
                    if vis_mask is not None:
                        snap.field_points["blue"] = np.where(
                            vis_mask[i][:, None], canon, -1.0
                        ).astype(np.float32)

                # homography
                if Hs[i] is None:
                    Hs[i] = np.eye(3, np.float32)
                snap.H_video2field = Hs[i]

                # CNN points
                if self.vis_cfg.show_detected_keypoints:
                    if vis_mask is not None:
                        snap.video_points["blue"] = np.where(
                            vis_mask[i][:, None], coords_np[i], -1.0
                        ).astype(np.float32)
                    else:
                        snap.field_points["blue"] = coords_np[i]

                if self.vis_cfg.show_projected_detections:
                    # project player feet by team
                    snap.field_points["yellow"] = self.warp_points_np(
                        Hs[i], ply_feet[i][m1]
                    ) # darker
                    snap.field_points["red"] = self.warp_points_np(
                        Hs[i], ply_feet[i][m0]
                    ) # lighter

                self.buffer.put(
                    VideoFrame(
                        frame_id=frame_ids[i].item(),
                        timestamp=timestamps[i].item(),
                        image=imgs[i],
                        annotations=snap,
                    )
                )

        self.phase = "done"

    @staticmethod
    def warp_points_np(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts.reshape(0, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float32))
        return dst.reshape(-1, 2).astype(np.float32)

    def _extract_feet(self, dets: List) -> List[np.ndarray]:
        return [
            np.stack(((b[:, 0] + b[:, 2]) * 0.5, b[:, 3]), 1).astype(np.float32)
            for b in (det.boxes for det in dets)
        ]

    def stop(self) -> None: self.running = False


if __name__ == "__main__":
    InferenceProcess().run()
