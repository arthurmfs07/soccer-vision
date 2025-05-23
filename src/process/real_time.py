import queue
import threading
import numpy as np
from pathlib import Path
from typing import Optional

from src.config                      import RealTimeConfig, DataConfig, VisualizationConfig
from src.main_config                 import MainConfig
from src.model.batch                 import DatasetLoader
from src.model.detect.objdetect      import ObjectDetector
from src.visual.visualizer           import Visualizer
from src.visual.field                import PitchConfig
from src.struct.shared_data          import SharedAnnotations
from src.process.inference           import InferenceProcess
from src.process.visualization       import VisualizationProcess

class RealTimeInference:
    """Real-time object detection and perspective homography pipeline."""

    def __init__(self, persp_model_path: Optional[str] = None):
        # real-time & data configs
        self.rt_cfg = RealTimeConfig()
        self.data_cfg = DataConfig()
        self.vis_cfg  = VisualizationConfig()

        self.persp_model_path = persp_model_path
        if self.persp_model_path:
            self.rt_cfg.annotation_gap = -1

        self.setup_paths()
        self.build_detector()
        self.build_dataloader()
        self.build_persp_model()
        self.build_visualizer()

    def setup_paths(self):
        base = Path(__file__).resolve().parents[2] / "data"
        self.yolo_path       = base / "10--models" / "yolov8_finetuned.pt"
        # video path can be read from DataConfig if needed

    def build_detector(self):
        self.detector = ObjectDetector(self.yolo_path, conf=self.rt_cfg.yolo_conf)

    def build_dataloader(self):
        loader = DatasetLoader(config=self.data_cfg, skip_sec=100*60)
        self.dataloader = loader.load()

    def build_persp_model(self):
        if not self.persp_model_path:
            self.model = None
            return

        mc = MainConfig()  # pull train_type & save_path
        from src.model.perspect.model import PerspectModel

        self.model = PerspectModel(
            train_type=mc.TRAIN_TYPE,
            device=self.rt_cfg.device
        ).to(self.rt_cfg.device)

        ckpt = Path(mc.SAVE_PATH)
        if ckpt.is_file():
            print(f"→ loading PerspectModel checkpoint from {ckpt}")
            self.model.load(str(ckpt))
        else:
            print(f"⚠️  No PerspectModel found at {ckpt}, starting from scratch")

        self.model.eval()

    def build_visualizer(self):
        blank = np.zeros(
            (self.data_cfg.height, self.data_cfg.width, 3),
            dtype=np.uint8
        )
        self.shared_data = SharedAnnotations()
        self.visualizer  = Visualizer(
            field_config=PitchConfig(),
            frame=blank,
            vis_config=self.vis_cfg,
            class_names=self.detector.class_names
        )

    def run(self):
        # Shared buffer
        buffer = queue.Queue(maxsize=self.rt_cfg.max_buffer_size)

        inference = InferenceProcess(
            detector=self.detector,
            dataloader=self.dataloader,
            buffer=buffer,
            config=self.rt_cfg,
            shared_data=self.shared_data,
            model=self.model
        )

        visualization = VisualizationProcess(
            buffer=buffer,
            visualizer=self.visualizer,
            config=self.rt_cfg,
            shared_data=self.shared_data
        )

        # link for drawing overlays
        self.visualizer.process = inference

        # run threads
        t_inf = threading.Thread(target=inference.process_batches)
        t_vis = threading.Thread(target=visualization.process_frames)
        t_inf.start()
        t_vis.start()

        try:
            t_inf.join()
            t_vis.join()
        except KeyboardInterrupt:
            inference.stop()
            visualization.stop()


if __name__ == "__main__":
    mc = MainConfig()
    pipeline = RealTimeInference(persp_model_path=mc.SAVE_PATH)
    pipeline.run()
