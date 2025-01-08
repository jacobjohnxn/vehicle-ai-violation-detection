from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO(r'models/newcombinedmodel.pt')  # Path to your trained weights

# Path to the input video
video_path = r'c:\Users\jacob\OneDrive\Desktop\project needs\testimages\livoback.mp4'

# Desired display height (change this value)
desired_height = 720

# Initialize video capture
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # Exit loop if no more frames

    # Run inference on the current frame
    results = model(frame)

    # Get the annotated frame as a NumPy array
    annotated_frame = results[0].plot()

    # Calculate new dimensions while maintaining aspect ratio
    height, width, _ = annotated_frame.shape
    aspect_ratio = width / height
    new_width = int(desired_height * aspect_ratio)
    resized_frame = cv2.resize(annotated_frame, (new_width, desired_height))

    # Display the resized frame
    cv2.imshow('Annotated Video', resized_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
