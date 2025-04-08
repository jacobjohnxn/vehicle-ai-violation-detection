!pip install easyocr ultralytics opencv-python-headless

# Import necessary libraries
import cv2
import numpy as np
import easyocr
import csv
import os
import time
from ultralytics import YOLO
from datetime import datetime
from google.colab import drive  # For mounting Google Drive
import torch  # Import PyTorch to check CUDA availability
import re
# Mount Google Drive
drive.mount('/content/drive')
# Define the base path in Google Drive (adjust this to your folder)
BASE_PATH = '/content/drive/MyDrive/vehicle_detection1/'

class VehicleDetectionSystem:
    def __init__(self):
        # Set up paths for saving results in Google Drive
        self.detected_vehicles_path = os.path.join(BASE_PATH, 'detected_vehicles')
        self.violation_frames_path = os.path.join(BASE_PATH, 'violation_frames')
        self.processed_videos_path = os.path.join(BASE_PATH, 'processed_videos')

        # Create all necessary directories
        os.makedirs(self.detected_vehicles_path, exist_ok=True)
        os.makedirs(self.violation_frames_path, exist_ok=True)
        os.makedirs(self.processed_videos_path, exist_ok=True)

        self.csv_file = os.path.join(BASE_PATH, 'vehicle_data.csv')
        self.initialize_csv()

        # Check for GPU availability and set the device
        self.device = 'cuda'
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Using device: {self.device}")

        # Initialize EasyOCR with GPU support if CUDA is available
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))

        # Initialize YOLO models (they automatically use the default device)
        self.custom_model = YOLO(os.path.join(BASE_PATH, 'best.pt'))  # Custom YOLO model
        self.bike_model = YOLO(os.path.join(BASE_PATH, 'yolov8n.pt'))     # Pre-trained YOLOv8n model

        # Verify the device the models are running on
        print(f"Custom model device: {next(self.custom_model.model.parameters()).device}")
        print(f"Bike model device: {next(self.bike_model.model.parameters()).device}")

        # Class mappings
        self.vehicle_classes = {2: 'car', 3: 'motorcycle'}
        self.class_names = {0: 'helmet', 1: 'no_helmet', 2: 'license_plate'}

        self.state_codes = ['KL', 'TN', 'AP', 'MH', 'KA', 'TS', 'DL', 'GJ', 'HP']
        self.processed_plates = {}
        self.last_violation = None
        self.last_violation_time = 0
        self.violation_cooldown = 2.0
        self.location_threshold = 50

    def initialize_csv(self):
        """Initialize the CSV file for logging detections if it doesn't exist."""
        header = ['Timestamp', 'Plate Number', 'Violation', 'Image Path']
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def preprocess_for_ocr(self, roi):
        """Preprocess the region of interest for OCR."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

    def save_detection(self, plate_text, violation, img_path):
        """Save detection data to the CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, plate_text, violation, img_path])

    def validate_state_plates(self, text):
        """Validate license plate format based on state-specific patterns."""
        import re
        patterns = {
            'KL': [r'^KL(0[1-9]|[1-9][0-9])[A-Z]\d{4}$'],
            'TN': [r'^TN\d{2}[A-Z]{1,2}\d{4}$'],
            'AP': [r'^AP\d{2}[A-Z]{1,2}\d{4}$'],
            'MH': [r'^MH\d{2}[A-Z]{1,2}\d{4}$'],
            'KA': [r'^KA\d{2}[A-Z]{1,2}\d{4}$'],
            'TS': [r'^TS\d{2}[A-Z]{1,2}\d{4}$'],
            'DL': [r'^DL\d{1,2}[A-Z]{1,2}\d{4}$'],
            'GJ': [r'^GJ\d{2}[A-Z]{1,2}\d{4}$'],
            'HP': [r'^HP\d{2}[A-Z]\d{4}$']
        }
        state_code = text[:2]
        if state_code in patterns:
            return any(bool(re.match(pattern, text)) for pattern in patterns[state_code])
        return False


    def process_plate_text(self, text):
        """Clean and validate the OCR-extracted plate text with proper formatting."""
        cleaned_text = ''.join(e for e in text if e.isalnum()).upper()

        # Ensure the state code exists in the predefined list
        if len(cleaned_text) < 8 or cleaned_text[:2] not in self.state_codes:
            return None

        state_code = cleaned_text[:2]
        rest = cleaned_text[2:]

        # Ensure the next two characters are exactly two digits
        match = re.match(r'(\d{1,2})([A-Z]+)(\d{4,5})$', rest)
        if match:
            district_code, letters, number = match.groups()

            # Ensure district code is always two digits (pad with zero if needed)
            district_code = district_code.zfill(2)

            formatted_plate = f"{state_code}{district_code}{letters}{number}"
            return formatted_plate if self.validate_state_plates(formatted_plate) else None

        return None


    def check_violations(self, frame):
        """Check for helmet and triple-riding violations using the specified device."""
        # Run YOLO models with the specified device
        custom_results = self.custom_model(frame, device=self.device)
        helmet_count = 0
        no_helmet_count = 0
        plates = []

        for result in custom_results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            x1, y1, x2, y2 = map(int, result[:4])

            if confidence > 0.15 and class_id == 2:  # License plate
                plates.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "License Plate", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        bike_results = self.bike_model(frame, device=self.device)
        bikes = []

        for result in bike_results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            if confidence > 0.3 and class_id == 3:  # Motorcycle
                x1, y1, x2, y2 = map(int, result[:4])
                bikes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Motorcycle", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if bikes:
            for bike in bikes:
                bike_x1, bike_y1, bike_x2, bike_y2 = bike
                bike_center_x = (bike_x1 + bike_x2) // 2
                bike_center_y = (bike_y1 + bike_y2) // 2
                bike_width = bike_x2 - bike_x1

                for result in custom_results[0].boxes.data:
                    class_id = int(result[5])
                    confidence = float(result[4])
                    x1, y1, x2, y2 = map(int, result[:4])
                    person_center_x = (x1 + x2) // 2
                    person_center_y = (y1 + y2) // 2

                    horizontal_threshold = bike_width * 1.5
                    vertical_threshold = bike_width

                    if (confidence > 0.25 and
                        abs(person_center_x - bike_center_x) < horizontal_threshold and
                        abs(person_center_y - bike_center_y) < vertical_threshold):

                        if class_id == 0:  # Helmet
                            helmet_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "Helmet", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif class_id == 1:  # No helmet
                            no_helmet_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "No Helmet", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            total_riders = helmet_count + no_helmet_count
            cv2.putText(frame, f"Total Riders: {total_riders}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            violations = []
            if no_helmet_count > 0:
                violations.append(f"Helmet violation - {no_helmet_count} rider(s)")
            if total_riders > 2:
                violations.append(f"Triple riding violation - {total_riders} riders")
                cv2.putText(frame, "Triple Riding Detected!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            violation = " & ".join(violations) if violations else "No violation"
            return violation, plates, (len(violations) > 0)

        return "No violation", plates, False

    def process_frame(self, frame):
        """Process a single frame for vehicle detection and violations."""
        current_time = time.time()
        violation, plates, is_violation = self.check_violations(frame)

        if is_violation and not plates:
            if (violation != self.last_violation or
                current_time - self.last_violation_time > self.violation_cooldown):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(self.violation_frames_path, f'frame_UNKNOWN_{timestamp}.jpg')
                print(f"Saving violation frame: {img_path}")
                cv2.imwrite(img_path, frame)
                print(f"Calling save_detection for UNKNOWN plate")
                self.save_detection("UNKNOWN", violation, img_path)
                self.last_violation = violation
                self.last_violation_time = current_time
                cv2.putText(frame, f"Violation: {violation}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for plate_coords in plates:
            x1, y1, x2, y2 = plate_coords
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            enhanced_roi = self.preprocess_for_ocr(roi)
            results = self.reader.readtext(enhanced_roi)

            if results:
                text = ' '.join([result[1] for result in results])
                cleaned_text = self.process_plate_text(text)

                if (cleaned_text and
                    (cleaned_text not in self.processed_plates or
                    current_time - self.processed_plates[cleaned_text] > 60)):
                    self.processed_plates[cleaned_text] = current_time

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_folder = self.violation_frames_path if is_violation else self.detected_vehicles_path
                    img_path = os.path.join(save_folder, f'frame_{cleaned_text}_{timestamp}.jpg')
                    cv2.imwrite(img_path, frame)
                    print(f"Calling save_detection for plate {cleaned_text}")
                    self.save_detection(cleaned_text, violation, img_path)

        return frame

    def run(self, video_path):
        """Run the vehicle detection system on a video file."""
        cap = cv2.VideoCapture(video_path)
        skip_rate = 10  # Process every 5th frame
        frame_count = 0  # Initialize frame counter

        TARGET_HEIGHT = 640
        print("Processing started...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no more frames

            frame_count += 1
            if frame_count % skip_rate != 0:
                continue  # Skip this frame

            height, width = frame.shape[:2]
            aspect_ratio = width / height
            new_width = int(TARGET_HEIGHT * aspect_ratio)
            frame = cv2.resize(frame, (new_width, TARGET_HEIGHT))

            processed_frame = self.process_frame(frame)
            # No display here - frames are processed and saved only

        cap.release()
        video_name = os.path.basename(video_path)
        processed_path = os.path.join(self.processed_videos_path, video_name)
        os.rename(video_path, processed_path)
        print(f"Video moved to: {processed_path}")
        print("Processing completed.")
if __name__ == "__main__":
    # Set the videos directory path in Google Drive
    videos_dir = '/content/drive/MyDrive/videos/'

    # Get all video files from the directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(video_extensions)]

    # Initialize the detection system
    system = VehicleDetectionSystem()

    # Process each video
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"\nProcessing video: {video_file}")
        system.run(video_path)
        print(f"Completed processing: {video_file}\n")

    print(f"Successfully processed {len(video_files)} videos!")