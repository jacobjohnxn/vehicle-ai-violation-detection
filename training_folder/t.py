import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLOv8 model
model_path = r'yolov8n.pt'  # Replace with your model path (e.g., yolov8m.pt, yolov8l.pt, yolov8x.pt)
model = YOLO(model_path)

# Initialize the camera (0 is the default webcam, use 1 or 2 if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously get frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform inference on the captured frame
    results = model(frame)

    # Parse the results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    class_names = results[0].names  # Class names

    # Draw boxes and labels on the frame
    for i, (box, conf, label) in enumerate(zip(boxes, confidences, labels)):
        x1, y1, x2, y2 = box
        label_name = class_names[int(label)]
        confidence = round(conf, 2)

        # Draw the bounding box and label on the image
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'{label_name} {confidence}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame with detections
    cv2.imshow('YOLOv8 Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
