import cv2
import numpy as np
import easyocr
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import os
import time
from ultralytics import YOLO

class VehicleDetectionSystem:
    def __init__(self):
        os.makedirs('detected_vehicles', exist_ok=True)
        os.makedirs('violation_frames', exist_ok=True)
        self.csv_file = 'rrealvehicle_data.csv'
        self.initialize_csv()
        
        # Initialize models
        self.reader = easyocr.Reader(['en'])
        # Single custom model for all detections
        self.custom_model = YOLO(r'models/newyolox.pt')
        self.bike_model = YOLO('yolov8n.pt')  # For general vehicle detection
        self.vehicle_classes = {
        2: 'car',
        3: 'motorcycle',
        }
        # Class mappings for the custom model
        self.class_names = {
            0: 'helmet',
            1: 'no_helmet',
            2: 'license_plate',

        }
        
        self.state_codes = ['KL', 'TN', 'AP', 'MH', 'KA', 'TS', 'DL', 'GJ', 'HP']
        self.processed_plates = {}
        self.plate_cache = self.load_plate_cache()
        self.last_violation = None
        self.last_violation_time = 0
        self.violation_cooldown = 2.0
        self.location_threshold = 50

    def load_plate_cache(self):
        plate_cache = {}
        if os.path.exists(self.csv_file):
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    plate_cache[row['Plate Number']] = row['Vehicle Model']
        return plate_cache

    def initialize_csv(self):
        header = ['Timestamp', 'Plate Number', 'Vehicle Model', 'Violation', 'Image Path']
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def preprocess_for_ocr(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

   

    def fetch_vehicle_model(self, plate_number):
        if plate_number in self.plate_cache:
            return self.plate_cache[plate_number]

        base_url = "https://www.carinfo.app/rc-details"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            formatted_plate = plate_number.replace(' ', '').upper()
            url = f"{base_url}/{formatted_plate}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Corrected element selection
            model_tag = soup.select_one("p.input_vehical_layout_vehicalModel__1ABTF")
            if model_tag:
                model_name = model_tag.text.strip()
                self.plate_cache[plate_number] = model_name
                return model_name

            return 'Unknown Model'

        except Exception as e:
            print("Error:", e)
            return 'Unknown Model'


    def save_detection(self, plate_text, model_name, violation, img_path):
        # Only save if there's a violation or if we successfully fetched vehicle details
        if violation != "No violation" or model_name != "Unknown Model":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, plate_text, model_name, violation, img_path])


    def validate_state_plates(self, text):
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
        cleaned_text = ''.join(e for e in text if e.isalnum()).upper()
        if len(cleaned_text) >= 8 and cleaned_text[:2] in self.state_codes:
            return cleaned_text
        return None

    def check_violations(self, frame):
        custom_results = self.custom_model(frame)
        helmet_count = 0
        no_helmet_count = 0
        plates = []
        
        # Process license plates first
        for result in custom_results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            x1, y1, x2, y2 = map(int, result[:4])
            
            if confidence > 0.15 and class_id == 2:  # license plate
                plates.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "License Plate", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Detect bikes
        bike_results = self.bike_model(frame)
        bikes = []
        
        for result in bike_results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            if confidence > 0.3 and class_id == 3:  # motorcycle
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
                    
                    # Expanded detection zone
                    horizontal_threshold = bike_width * 1.5
                    vertical_threshold = bike_width  # Using bike width as reference
                    
                    if (confidence > 0.25 and 
                        abs(person_center_x - bike_center_x) < horizontal_threshold and 
                        abs(person_center_y - bike_center_y) < vertical_threshold):
                        
                        if class_id == 0:  # helmet
                            helmet_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Helmet", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif class_id == 1:  # no helmet
                            no_helmet_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"No Helmet", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            total_riders = helmet_count + no_helmet_count
            cv2.putText(frame, f"Total Riders: {total_riders}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            violations = []
            if no_helmet_count > 0:
                violations.append(f"Helmet violation - {no_helmet_count} rider(s)")
            if total_riders > 2:
                violations.append(f"Triple riding violation - {total_riders} riders")
                cv2.putText(frame, f"Triple Riding Detected!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            violation = " & ".join(violations) if violations else "No violation"
            return violation, plates, (len(violations) > 0)
        
        return "No violation", plates, False





    def process_frame(self, frame):
        current_time = time.time()
        violation, plates, is_violation = self.check_violations(frame)
        
        # For unknown violations (no plate), use cooldown
        if is_violation and not plates:
            # Check if enough time has passed since last violation
            if (violation != self.last_violation or 
                current_time - self.last_violation_time > self.violation_cooldown):
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f'violation_frames/frame_UNKNOWN_{timestamp}.jpg'
                cv2.imwrite(img_path, frame)
                self.save_detection("UNKNOWN", "Unknown Model", violation, img_path)
                
                # Update tracking
                self.last_violation = violation
                self.last_violation_time = current_time
                
                # Display violation on frame
                cv2.putText(frame, f"Violation: {violation}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Process plates with cooldown
        for plate_coords in plates:
            x1, y1, x2, y2 = plate_coords
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Preprocess ROI for OCR
            enhanced_roi = self.preprocess_for_ocr(roi)

            # Read text using EasyOCR
            results = self.reader.readtext(enhanced_roi)

            # Debugging: Print all OCR results
            print("Raw OCR Results:", results)

            if results:
                extracted_text = ' '.join([result[1] for result in results])
                cleaned_text = self.process_plate_text(extracted_text)

                # Debugging: Print cleaned text
                print(f"Extracted Plate: {extracted_text} -> Cleaned Plate: {cleaned_text}")

                if cleaned_text:
                    # Ensure it follows the state validation rules
                    if self.validate_state_plates(cleaned_text):
                        print(f"Valid Plate Number: {cleaned_text}")
                    else:
                        print(f"Invalid Plate Format: {cleaned_text}")

                    # Save detected plate in CSV
                    model_name = self.fetch_vehicle_model(cleaned_text)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = f'detected_vehicles/frame_{cleaned_text}_{timestamp}.jpg'
                    cv2.imwrite(img_path, frame)
                    self.save_detection(cleaned_text, model_name, violation, img_path)

        
        return frame


    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return
        
        # Set desired vertical height
        TARGET_HEIGHT = 720  # Adjust as needed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get original dimensions
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            new_width = int(TARGET_HEIGHT * aspect_ratio)
            
            # Resize the frame while maintaining aspect ratio
            frame = cv2.resize(frame, (new_width, TARGET_HEIGHT))
            
            # Process the frame (implement actual processing logic in process_frame)
            processed_frame = self.process_frame(frame)
            
            # Display the processed frame
            cv2.imshow('Vehicle Detection System', processed_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = VehicleDetectionSystem()
    system.run(r'c:\Users\jacob\OneDrive\Desktop\project needs\testimages\dio-bs6-modification-viral-video-views-kerala-kasaragod-wayanad-ernakulam-dio-modified-720-ytshorts.savetube.me.mp4')
    #system.run(r'c:\Users\jacob\OneDrive\Desktop\project needs\testimages\sonalmiss.mp4')

