from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO('yolov8m.pt')
    results = model.train(
        data="data/00--raw/football-players-detection.v12i.yolov8/data.yaml",
        imgsz=640,
        batch=8,
        device=0,
        epochs=300,
        patience=30,
        amp=False,
        mosaic=0.0,
        plots=True,
        save=True,
        seed=0,
        deterministic=True,
        name='yolov8m-detect-imgsz640-300ep-30pat'
    )