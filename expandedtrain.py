from ultralytics import YOLO
import torch
import os
import yaml

def find_detection_head(model):
    """Find the detection head layer in a YOLOv8 model"""
    for name, module in model.named_modules():
        # The detection head in YOLOv8 is an instance of the Detect class
        if module.__class__.__name__ == 'Detect':
            return name
    return None

def verify_yaml_file(yaml_path):
    """Verify that the YAML file exists and has the expected structure"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"YAML file missing required key: {key}")
        
        # Check number of classes
        if not isinstance(data['nc'], int) or data['nc'] <= 0:
            raise ValueError(f"Invalid number of classes: {data['nc']}")
            
        # Check class names
        if len(data['names']) != data['nc']:
            raise ValueError(f"Mismatch between nc ({data['nc']}) and number of names ({len(data['names'])})")
            
        return data['nc']
    except yaml.YAMLError:
        raise ValueError(f"Invalid YAML format in file: {yaml_path}")

if __name__ == '__main__':
    try:
        # Fixed parameters (no command line options)
        ckpt_path = 'saved/yolov8_uec_food100/weights/best.pt'
        data_yaml_path = 'data/UEC_Food_256/yolo/data.yaml'
        base_model = 'yolov8s.pt'
        experiment_name = 'yolov8_uec_food256_expanded'
        
        # Verify checkpoint exists
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        # Verify data YAML file and get number of classes
        nc = verify_yaml_file(data_yaml_path)
        print(f"Training on dataset with {nc} classes")

        # Check if CUDA is available and set the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize a new model from base weights
        model = YOLO(base_model)
        
        # Find the detection head automatically
        detection_head = find_detection_head(model.model)
        if not detection_head:
            raise ValueError("Could not find detection head in the model!")
        print(f"Found detection head at: {detection_head}")
        
        # Load weights from the 100-class checkpoint to the appropriate device
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Check if it's a valid checkpoint
        if 'model' not in checkpoint:
            raise ValueError(f"Invalid checkpoint format: 'model' not found in {ckpt_path}")
            
        # Transfer all weights except the detection head
        model.model.load_state_dict({k: v for k, v in checkpoint['model'].state_dict().items() 
                                if detection_head not in k}, strict=False)
        
        print(f"Model loaded with weights transferred from checkpoint (excluding {detection_head})")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join('saved', experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Train the model on the 256-class dataset
        results = model.train(
            data=data_yaml_path,              # Path to dataset YAML
            epochs=30,                        # Number of epochs
            imgsz=640,                        # Image size
            batch=16,                         # Batch size
            project='saved',                  # Project directory
            name=experiment_name,             # Run name
            exist_ok=True,                    # Overwrite existing project
            patience=2,                      # Early stopping patience
            save_period=5,                    # Save checkpoint every 5 epochs
            lr0=0.01,                         # Initial learning rate
            lrf=0.001,                        # Final learning rate
            augment=True,                     # Use data augmentation
            verbose=True,                     # Verbose output
            resume=False,                     # Don't resume from last.pt
            device=0 if device.type == 'cuda' else 'cpu',  # Use GPU if available
            workers=8,                        # Number of dataloader workers
            cos_lr=True,                      # Use cosine LR scheduler
            close_mosaic=5,                   # Disable mosaic augmentation for final epochs
        )
        
        print(f"Training complete! Model saved to {output_dir}/")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()