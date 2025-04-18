from ultralytics import YOLO

if __name__ == '__main__':
    # Load the last checkpoint
    model = YOLO('C:/Intern/FoodApp/saved/yolov8_uec_food100/weights/last.pt')

    # Continue training with enhanced configurations
    model.train(
        data='C:/Intern/FoodApp/data/UEC_Food_100/yolo/data.yaml',
        epochs=10,          # Increased for more training
        imgsz=640,          # Option: increase to 1280 if resources allow
        batch=16,           # Increased, adjust based on GPU memory
        project='C:/Intern/FoodApp/saved',
        name='yolov8_uec_food100',
        exist_ok=True,
        augment=True,       # Enable data augmentation
        lr0=0.01,           # Initial learning rate
        lrf=0.001,          # Final learning rate for scheduler
        patience=10,        # Early stopping if no improvement
        save_period=5,      # Save checkpoints every 5 epochs
    )