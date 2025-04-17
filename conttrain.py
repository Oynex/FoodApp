from ultralytics import YOLO

if __name__ == '__main__':
    # Load the last checkpoint
    model = YOLO('saved/yolov8_uec_food100/weights/last.pt')

    # Continue training
    model.train(
        data=r'data/UEC_Food_100/yolo/data.yaml',
        epochs=1,  # Set to the number of additional epochs you want
        imgsz=640,
        batch=10,
        project='saved',
        name='yolov8_uec_food100',
        exist_ok=True
    )
