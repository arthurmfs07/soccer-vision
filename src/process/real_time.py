import torch
import queue
import threading
import numpy as np
from pathlib import Path
from typing import Optional

from src.model.batch import DatasetLoader
from src.model.detect.objdetect import ObjectDetector

from src.visual.visualizer import Visualizer
from src.visual.field import PitchConfig

from src.struct.shared_data import SharedAnnotations

from src.config import RealTimeConfig, DataConfig
from src.process.inference import InferenceProcess
from src.process.visualization import VisualizationProcess

class RealTimeInference:

    """Real-time object detection and visualization pipeline."""

    def __init__(self, persp_model_path: Optional[str] = None):
        
        self.config = RealTimeConfig()
        self.persp_model_path = persp_model_path
        if self.persp_model_path:
            self.config.annotation_gap = -1
        self.batch_size = self.config.batch_size
        self.max_buffer_size = self.config.max_buffer_size
        self.annotation_gap = self.config.annotation_gap
        self.setup()


    def setup(self):
        self.data_path = Path(__file__).resolve().parents[2] / "data"
        self.yolo_path = self.data_path/ "10--models" / "yolov8_finetuned.pt"
        game_name = "JOGO COMPLETO： WERDER BREMEN X BAYERN DE MUNIQUE ｜ RODADA 1 ｜ BUNDESLIGA 23⧸24.mp4"
        self.example_video_path = self.data_path / "00--raw" / "videos" / game_name
        

    def run(self):
        # Load YOLO detector
        detector = ObjectDetector(self.yolo_path, conf=RealTimeConfig.yolo_conf)

        # Load dataset
        dataconfig = DataConfig()
        dataset = DatasetLoader(config=dataconfig, skip_sec=100*60)
        dataloader = dataset.load()

        # Shared buffer (FIFO queue)
        buffer = queue.Queue(maxsize=RealTimeConfig().max_buffer_size)  # Adjust buffer size if needed

        shared_data = SharedAnnotations()

        cnn_model = None
        if self.persp_model_path:
            from src.model.perspect.cnn import CNN
            cnn_model = CNN().to(self.config.device)
            cnn_model.load_state_dict(
                torch.load(self.persp_model_path, map_location=self.config.device)
            )
            cnn_model.eval()

        inference_process = InferenceProcess(
            detector, 
            dataloader, 
            buffer,
            config=self.config,
            shared_data=shared_data,
            cnn_model=cnn_model
            )


        visualizer = Visualizer(
            PitchConfig(), 
                np.zeros(
                    (dataconfig.width, dataconfig.height, 3), 
                    dtype=np.uint8), 
                detector.class_names,
                process=inference_process
                )
        visualization = VisualizationProcess(
            buffer, visualizer, self.config, 
            shared_data=shared_data)

        # Start threads
        inference_thread = threading.Thread(target=inference_process.process_batches)
        visualization_thread = threading.Thread(target=visualization.process_frames)

        inference_thread.start()
        visualization_thread.start()

        try:
            inference_thread.join()
            visualization_thread.join()
        except KeyboardInterrupt:
            inference_process.stop()
            visualization.stop()


if __name__ == "__main__":

    pipeline = RealTimeInference(
        persp_model_path="data/10--models/perspect_cnn.pth"
        )
    pipeline.run()