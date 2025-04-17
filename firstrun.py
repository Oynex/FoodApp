from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8s model and force it to use CUDA
    model = YOLO('C:/Intern/FoodApp/yolov8s.pt')

    # Train the model
    model.train(
        data='C:/Intern/FoodApp/data/UEC_Food_100/yolo/data.yaml',
        epochs=1,                # You can adjust the number of epochs
        imgsz=640,               # Image size (default 640)
        batch=10,                # Batch size (adjust based on your GPU)
        project='C:/Intern/FoodApp/saved',         # Folder to save results
        name='yolov8_uec_food100', # Subfolder name for this run
        exist_ok=True            # Overwrite if folder exists
    )
