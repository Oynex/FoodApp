from ultralytics import YOLO

# Load the YOLOv8s model
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data=r'f:/REPO/FoodApp/data/UEC_Food_100/yolo/data.yaml',
    epochs=100,                # You can adjust the number of epochs
    imgsz=640,                 # Image size (default 640)
    batch=16,                  # Batch size (adjust based on your GPU)
    project='saved',           # Folder to save results
    name='yolov8_uec_food100', # Subfolder name for this run
    exist_ok=True              # Overwrite if folder exists
)
