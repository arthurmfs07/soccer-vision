import argparse
import torch
import numpy as np
from typing import Optional

from src.process.process import Process
from src.model.perspect.train_sq import PerspectSquareTrainer, SquareTrainConfig
from src.model.perspect.train_pt import PerspectPointsTrainer, PointsTrainConfig
from src.visual.visualizer import Visualizer
from src.struct.shared_data import SharedAnnotations
from src.struct.utils import create_base_square
from src.visual.video import VideoFrame
from src.visual.field import PitchConfig
from src.config import VisualizationConfig


class TrainerProcess(Process):
    """Wraps the training process to allow real‐time visualization."""

    def __init__(
        self,
        trainer,
        visualizer: Visualizer,
        dispaly_scale: float = 0.5,
        shared_data: Optional[SharedAnnotations] = SharedAnnotations(),
    ) -> None:
        self.trainer       = trainer
        self.display_scale = dispaly_scale
        self.visualizer    = visualizer
        self.running       = False
        self.shared_data   = shared_data

    def on_mouse_click(self, x: int, y: int) -> None:
        pass

    def is_done(self) -> bool:
        return not self.running

    def stop(self) -> None:
        self.running = False

    def run_rendered_train(self, epochs: int = -1) -> None:
        """Runs trainer.train_batches(), updates Visualizer each batch."""
        self.running = True

        def on_batch(
            phase: str, 
            batch_idx: int, 
            images: torch.Tensor, 
            predictions: torch.Tensor, 
            gt_pts: torch.Tensor,
            loss_val: float
        ):
            if not self.running:
                raise StopIteration()

            if phase == "homography":
                # square‐mode: gt_pts & predictions are [B,4,2]
                self.shared_data.video_points["yellow"] = create_base_square(as_tensor=False).tolist()
                self.shared_data.field_points["yellow"] = [tuple(pt.tolist()) for pt in gt_pts[0]]
                self.shared_data.field_points["blue"]   = predictions[0].tolist()

            elif phase == "points":
                # points‐mode: predictions [B,N,3] = (x,y,vis_logit)
                gt_arr   = gt_pts[0].detach().cpu().numpy() # [32,3]
                gt_xy    = gt_arr[:, :2].astype(np.float32)
                mask_gt  = gt_arr[:, 2] > 0
                gt_video = np.where(mask_gt[:, None], gt_xy, -1.0)

                pred_arr   = predictions[0].detach().cpu().numpy() # [[x,y,logits],...]
                pred_xy    = pred_arr[:, :2].astype(np.float32)
                mask_pred  = pred_arr[:, 2] > 0
                pred_video = np.where(mask_pred[:, None], pred_xy, -1.0)


                self.shared_data.numbered_video_points["yellow"] = gt_video
                self.shared_data.numbered_video_points["blue"]   = pred_video
            


            video_frame = VideoFrame(
                frame_id=batch_idx,
                timestamp=0,
                image=self.prepare_image(images[0].clone()),
                annotations=None
            )
            self.visualizer.video_vis.update(video_frame)
            self.visualizer.render()

        try:
            self.trainer.train(epochs=epochs, on_batch=on_batch)
        except StopIteration:
            self.logger.info("Training interrupted by visualizer stop.")
        finally:
            self.running = False

    def prepare_image(self, img: torch.Tensor) -> np.ndarray:
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return img_np



if __name__ == "__main__":
    from src.main_config import MainConfig
    mc = MainConfig()

    if mc.TRAIN_TYPE == "square":
        cfg     = SquareTrainConfig(
            dataset_folder=mc.DATASET_PATH,
            batch_size=mc.BATCH_SIZE,
            lr=mc.LR,
            patience=mc.PATIENCE,
            device=mc.DEVICE,
            save_path=mc.SAVE_PATH,
        )
        trainer = PerspectSquareTrainer(cfg)
    else:
        cfg     = PointsTrainConfig(
            dataset_folder=mc.DATASET_PATH,
            batch_size=mc.BATCH_SIZE,
            lr=mc.LR,
            patience=mc.PATIENCE,
            device=mc.DEVICE,
            save_path=mc.SAVE_PATH,
        )
        trainer = PerspectPointsTrainer(cfg)

    # set up visualization
    dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
    shared_data = SharedAnnotations()
    visualizer  = Visualizer(
        field_config=PitchConfig(),
        frame=dummy_frame,
        vis_config=VisualizationConfig()
    )

    # launch training‐with‐viz
    trainer_process = TrainerProcess(
        trainer=trainer,
        visualizer=visualizer,
        shared_data=shared_data
    )
    visualizer.process = trainer_process
    trainer_process.run_rendered_train()
