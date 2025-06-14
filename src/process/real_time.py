import queue
import threading
import numpy as np
from pathlib import Path
from typing import Optional

from src.config                      import MainConfig, RealTimeConfig, VisualizationConfig
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
        self.main_cfg = MainConfig()
        self.rt_cfg = RealTimeConfig()
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
        self.detect_yolo_path = self.main_cfg.yolo_paths["detect"]
        self.keypoints_yolo_path = self.main_cfg.yolo_paths["keypoints"]

    def build_detector(self):
        self.detector = ObjectDetector(
            self.detect_yolo_path, 
            conf=self.rt_cfg.detect_conf
            )

    def build_dataloader(self):
        loader = DatasetLoader(config=self.rt_cfg)
        self.dataloader = loader.load()


    def build_persp_model(self):
        if not self.persp_model_path:
            self.model = None
            return

        mc = self.main_cfg


        # Handcrafted model if needed
        # self.model = self._build_handcrafted_model()

        from src.model.perspect.yolo_model import YOLOModel
        self.model = YOLOModel(
            self.keypoints_yolo_path, 
            imgsz=self.rt_cfg.imgsz, 
            device=self.rt_cfg.device
            )


    def build_visualizer(self):
        blank = np.zeros(
            (self.rt_cfg.imgsz, self.rt_cfg.imgsz, 3),
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
        self.visualizer.process = visualization

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


    def _build_handcrafted_model(self):
    
        mc = self.main_cfg
        from src.model.perspect.handcraft.model import build_model, BasePerspectModel

        self.model = build_model(model_type=mc.TRAIN_TYPE, device=self.rt_cfg.device)
        
        ckpt = Path(mc.SAVE_PATH)
        if ckpt.is_file():
            print(f"→ loading PerspectModel checkpoint from {ckpt}")
            self.model, _ = BasePerspectModel.load_checkpoint(
                str(ckpt), 
                device=self.rt_cfg.device
                )
        else:
            print(f"⚠️  No PerspectModel found at {ckpt}, starting from scratch")

        self.model.eval()



if __name__ == "__main__":
    mc = MainConfig()
    pipeline = RealTimeInference(persp_model_path=mc.SAVE_PATH)
    pipeline.run()
