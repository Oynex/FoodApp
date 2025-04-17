from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model('F:\REPO\FoodApp\data\image.jpg')
results[0].show()