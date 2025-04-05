import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.process.process import Process
from src.model.detect.finetune import YOLOTrainer
from src.model.perspect.train import PerspectTrainer
from src.visual.visualizer import Visualizer
from src.struct.shared_data import SharedAnnotations
from src.visual.video import VideoFrame


class TrainerProcess(Process):

    """Wraps Training Process to allow real-time visualization."""


    def __init__(
            self, 
            trainer: PerspectTrainer,
            visualizer: Visualizer, 
            dispaly_scale: float = 0.5, 
            shared_data: Optional[SharedAnnotations] = SharedAnnotations()
            ):
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
        """Return whether the process is finished."""
        return not self.running

    def stop(self):
        """Stop the process externally."""
        self.running = False



    def run_rendered_train(self, epochs: int = -1):
        """
        Runs training via trainer.train_batches() while updating the Visualizer in real time.
        Press 'q' to quit early.
        """
        train_iter = self.trainer.train_batches(epochs=epochs)
        self.running = True

        for (epoch_idx, batch_idx, images, labels, detections, yhat_field, loss_val) in train_iter:
            
            # yolo detections
            detection_pts = []
            for box in detections[0].boxes:
                x1, y1, x2, y2 = box
                foot = ((x1 + x2) / 2, y2)
                detection_pts.append( (float(foot[0]), float(foot[1])) )
            self.shared_data.video_detection_pts = detection_pts

            projected_pts = []
            pts0, pts1 = 0, 0
            for pt in yhat_field[0]:
                if pt[0] >= 0 and pt[1] >= 0:
                    projected_pts.append( (float(pt[0].item()), float(pt[1].item())) )
                    pts0 += pt[0]
                    pts1 += pt[1]
            self.shared_data.projected_detection_model_pts = projected_pts

            self.shared_data.ground_truth_pts = [tuple(pt) for pt in labels[0].tolist()]


            video_frame = VideoFrame(
                frame_id=batch_idx,
                timestamp=0,
                image=self.prepare_image(images[0].clone()),
                detections=[detections[0]]
            )

            self.visualizer.video_visualizer.update(video_frame)
            
            self.visualizer.render()

            if not self.running:
                break
            
        self.running = False
        print("Rendered training finished.")


    def prepare_image(self, img: torch.Tensor) -> np.ndarray:
        """Prepare for render
        input: tensor [C, W, H]
        """
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return img_np



    def finetune_yolo(self):
        # This method can call YOLOTrainer finetuning if desired.
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
    from src.model.perspect.train import PerspectTrainer
    from src.struct.shared_data import SharedAnnotations
    from src.visual.visualizer import Visualizer
    from src.visual.field import PitchConfig

    trainer = PerspectTrainer()

    dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
    shared_data = SharedAnnotations()

    visualizer = Visualizer(PitchConfig(), dummy_frame)

    trainer_process = TrainerProcess(
        trainer=trainer, 
        visualizer=visualizer, 
        shared_data=shared_data
        )

    visualizer.process = trainer_process


    trainer_process.run_rendered_train(epochs=1000)