import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.process.process import Process
from src.model.detect.finetune import YOLOTrainer
from src.model.perspect.train import PerspectTrainer
from src.visual.visualizer import Visualizer
from src.struct.shared_data import SharedAnnotations
from src.struct.utils import create_base_square
from src.visual.video import VideoFrame

class TrainerProcess(Process):
    """Wraps the training process to allow real‐time visualization for both homography and player phases."""
    def __init__(
        self,
        trainer: PerspectTrainer,
        visualizer: Visualizer,
        dispaly_scale: float = 0.5,
        shared_data: Optional[SharedAnnotations] = SharedAnnotations(),
    ) -> None:
        self.trainer = trainer
        self.display_scale = dispaly_scale
        self.visualizer = visualizer
        self.running = False
        self.window_name = self.visualizer.window_name
        self.shared_data = shared_data

    def on_mouse_click(self, x: int, y: int) -> None:
        """Not used in this example, but required by Process."""
        pass

    def is_done(self) -> bool:
        """Returns whether the process is finished."""
        return not self.running

    def stop(self) -> None:
        """Stops the process externally."""
        self.running = False

    def run_rendered_train(self, epochs: int = -1) -> None:
        """
        Runs training using trainer.train_batches() while updating the Visualizer in real time.
        Each batch yields a six-tuple: (epoch, phase, batch_idx, images, predictions, loss_val).
        The phase indicator ("homography" or "player") is used to update shared data for visualization.
        """
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
                raise StopIteration
            
            if phase == "homography":
                # For homography training, predictions contains the predicted square (as 4 corner points).
                self.shared_data.projected_detection_model_pts = predictions[0].tolist()
                self.shared_data.ground_truth_pts = [tuple(pt.tolist()) for pt in gt_pts[0]]
                self.shared_data.sampled_video_pts = create_base_square(images.shape, as_tensor=False).tolist()
            elif phase == "player":
                # For player position training, predictions contains the projected player positions.
                self.shared_data.projected_detection_pts = predictions[0].tolist()

            # Create a VideoFrame to update the visualizer – here we simply display the first image.
            video_frame = VideoFrame(
                frame_id=batch_idx,
                timestamp=0,
                image=self.prepare_image(images[0].clone()),
                detections=[]
            )
            self.visualizer.video_visualizer.update(video_frame)
            self.visualizer.render()
        try:
            self.trainer.train(epochs=epochs, on_batch=on_batch)
        except StopIteration:
            self.logger.info("Training interrupted by visualizer stop.")
        finally:
            self.running = False


    def prepare_image(self, img: torch.Tensor) -> np.ndarray:
        """
        Prepare an image for rendering.
        Converts a tensor of shape [C, H, W] to a uint8 numpy array of shape [H, W, 3].
        """
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return img_np

    def finetune_yolo(self) -> None:
        """
        Optional method to finetune YOLO.
        Calls the YOLOTrainer for detection finetuning.
        """
        from src.model.detect.finetune import YOLOTrainer
        data_path = Path(__file__).resolve().parents[3] / "data"
        dataset_yaml = data_path / "00--raw" / "football-players-detection.v12i.yolov8" / "data.yaml"
        trainer = YOLOTrainer(dataset_yaml, model_size="yolov8m", epochs=200, batch_size=16)
        save_path = data_path / "10--models" / "yolo_finetune2.pt"
        trainer.train()
        trainer.evaluate()
        trainer.save_model(save_path=save_path)

if __name__ == "__main__":
    import numpy as np
    from src.model.perspect.train import PerspectTrainer, TrainConfig
    from src.struct.shared_data import SharedAnnotations
    from src.visual.visualizer import Visualizer
    from src.visual.field import PitchConfig

    config = TrainConfig(
        dataset_folder=Path("data/annotated_homographies"),
        batch_size=16,
        lr=1e-5,
        patience=100,
        warp_alpha=0.05,
        device="cuda",
        save_path=Path("data/10--models/perspect_cnn2.pth"),
        train_on_homography=True,
        train_on_player_position=False  # Set to True if player position labels are available
    )

    trainer = PerspectTrainer(config)
    dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
    shared_data = SharedAnnotations()
    visualizer = Visualizer(PitchConfig(), dummy_frame)
    trainer_process = TrainerProcess(
        trainer=trainer,
        visualizer=visualizer,
        shared_data=shared_data
    )
    visualizer.process = trainer_process
    trainer_process.run_rendered_train()#epochs=1000)
    
