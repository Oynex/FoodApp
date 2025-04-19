from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Print the model structure
print("YOLOv8 model structure:")
print(model.model)

# Get model info for specific detection head layer
print("\nDetection head (model.24) details:")
if hasattr(model.model, 'model') and len(model.model.model) >= 25:
    print(f"Layer 24 type: {type(model.model.model[24])}")
    print(f"Layer 24 structure: {model.model.model[24]}")
else:
    print("Model structure is different than expected. Full model:")
    for i, layer in enumerate(model.model.model):
        print(f"Layer {i}: {type(layer).__name__}")