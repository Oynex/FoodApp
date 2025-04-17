from ultralytics import YOLO

if __name__ == '__main__':
    # Load the last checkpoint
    model = YOLO('C:/Intern/FoodApp/saved/yolov8_uec_food100/weights/last.pt')

    # Continue training
    model.train(
        data='C:/Intern/FoodApp/data/UEC_Food_100/yolo/data.yaml',
        epochs=10,  # Set to the number of additional epochs you want
        imgsz=640,
        batch=10,
        project='C:/Intern/FoodApp/saved',
        name='yolov8_uec_food100',
        exist_ok=True
    )
