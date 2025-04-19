from ultralytics import YOLO
import torch
import os
import yaml

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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

def find_optimal_batch_size(model, data_yaml_path, imgsz, start_batch=64, min_batch=4):
    """
    Find the optimal batch size that maximizes GPU memory usage without OOM errors
    
    Args:
        model: YOLO model instance
        data_yaml_path: Path to data.yaml file
        imgsz: Image size for training
        start_batch: Initial batch size to try
        min_batch: Minimum acceptable batch size
        
    Returns:
        Optimal batch size
    """
    print("Finding optimal batch size for maximum GPU utilization...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, using default batch size")
        return 16  # Default for CPU
    
    batch_size = start_batch
    
    # Try binary search approach for finding optimal batch size
    max_batch = start_batch
    min_search = min_batch
    optimal_batch = None
    
    while min_search <= max_batch:
        try:
            current_batch = (min_search + max_batch) // 2
            print(f"Trying batch size: {current_batch}")
            
            # Create a temporary test model
            test_model = model.model.deepcopy()
            
            # Try to run a single training step with the current batch size
            # We'll use a very small subset and just 1 iteration to test memory usage
            torch.cuda.empty_cache()  # Clear GPU memory
            
            # Use a dummy training run with just one iteration to test memory
            model.train(
                data=data_yaml_path,
                epochs=1,
                imgsz=imgsz,
                batch=current_batch,
                close_mosaic=0,  # Don't close mosaic
                exist_ok=True,
                verbose=False,
                device=0,
                fraction=0.01,  # Use just a tiny fraction of data for the test
                freeze=[0, 1, 2],  # Freeze some layers to save memory in the test
                val=False,  # Skip validation
                nbs=64,  # Nominal batch size for scaling
                _callbacks=None,  # No callbacks to speed up test
            )
            
            # If we got here without an OOM error, try a larger batch size
            optimal_batch = current_batch
            min_search = current_batch + 1
            print(f"Batch size {current_batch} works, trying larger...")
            
        except torch.cuda.OutOfMemoryError:
            # If we got an OOM error, try a smaller batch size
            print(f"Batch size {current_batch} caused OOM error, trying smaller...")
            max_batch = current_batch - 1
            torch.cuda.empty_cache()  # Clear GPU memory
            
        except Exception as e:
            # For other errors, try a smaller batch size as well
            print(f"Error with batch size {current_batch}: {str(e)}")
            max_batch = current_batch - 1
            torch.cuda.empty_cache()  # Clear GPU memory
    
    # If we couldn't find a working batch size, use the minimum
    if optimal_batch is None:
        optimal_batch = min_batch
        print(f"Could not find optimal batch size, using minimum: {min_batch}")
    else:
        print(f"Found optimal batch size: {optimal_batch}")
    
    return optimal_batch

if __name__ == '__main__':
    try:
        # Fixed parameters with absolute paths
        ckpt_path = os.path.join(PROJECT_ROOT, 'saved', 'yolov8_uec_food100', 'weights', 'best.pt')
        data_yaml_path = os.path.join(PROJECT_ROOT, 'data', 'UEC_Food_256', 'yolo', 'data.yaml')
        base_model = os.path.join(PROJECT_ROOT, 'yolov8s.pt')
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
        output_dir = os.path.join(PROJECT_ROOT, 'saved', experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Find the optimal batch size
        imgsz = 640
        optimal_batch = find_optimal_batch_size(model, data_yaml_path, imgsz, start_batch=64, min_batch=4)
        print(f"Training with dynamically determined optimal batch size: {optimal_batch}")
        
        # Train the model on the 256-class dataset
        results = model.train(
            data=data_yaml_path,              # Path to dataset YAML
            epochs=1,                          # Number of epochs
            imgsz=imgsz,                      # Image size
            batch=optimal_batch,              # Use dynamically determined batch size
            project=os.path.join(PROJECT_ROOT, 'saved'),  # Project directory (absolute path)
            name=experiment_name,             # Run name
            exist_ok=True,                    # Overwrite existing project
            patience=2,                       # Early stopping patience
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