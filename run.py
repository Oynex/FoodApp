from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Load the trained model
    model = YOLO('saved/yolov8_uec_food100/weights/best.pt')

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Failed to capture frame.')
            break

        # Run YOLO detection
        results = model(frame)

        # Visualize results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
