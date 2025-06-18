from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolov8l-pose.pt')

    results = model.train(
        data='data/00--raw/football-field-detection.v15i.yolov8/data.yaml',
        imgsz=640,
        batch=4,
        device=0,
        epochs=500,
        patience=30,
        amp=False,
        mosaic=0.0,
        plots=True,
        save=True,
        seed=0,
        deterministic=True,
        name='yolov8l-pose-imgsz640-500ep-30pat'
    )