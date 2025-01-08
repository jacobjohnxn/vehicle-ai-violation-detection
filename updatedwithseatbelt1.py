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
        self.csv_file = 'vehicle_data.csv'
        self.initialize_csv()
        
        # Initialize models with EasyOCR
        self.reader = easyocr.Reader(['en'])
        self.plate_model = YOLO(r'models/trainedindianplate.pt')
        self.vehicle_model = YOLO('yolov8n.pt')
        self.helmet_model = YOLO('yolov8n.pt')
        self.seatbelt_model = YOLO(r'models/seatbelt.pt')

        self.state_codes = ['KL', 'TN', 'AP', 'MH', 'KA', 'TS', 'DL', 'GJ', 'HP']
        self.processed_plates = {}
        self.plate_cache = self.load_plate_cache()

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
            print(f"Found {plate_number} in cache")
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
            
            model_div = soup.find('div', {'class': 'css-1yuhvjn'})
            if model_div:
                model_name = model_div.find('p', {'class': 'css-1s1bvzd'}).text.strip()
                self.plate_cache[plate_number] = model_name
                print(f"Fetched new model for {plate_number}")
                return model_name
            return 'Unknown Model'
        except Exception as e:
            print(f"Model fetch status: {str(e)}")
            return 'Unknown Model'

    def save_detection(self, plate_text, model_name, violation, img_path):
        # Remove the Unknown Model check to allow UNKNOWN entries
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, plate_text, model_name, violation, img_path])


    def draw_info(self, frame, x, y, w, h, text, model_name, violation):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Plate: {text}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Model: {model_name}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Violation: {violation}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    
    def validate_state_plates(self, text):
        import re
        patterns = {
            'KL': [r'^KL\d{2}[A-Z]{2}\d{4}$', r'^KL\d{2}[A-Z]\d{4}$'],  # Kerala
            'TN': [r'^TN\d{2}[A-Z]{1,2}\d{4}$'],  # Tamil Nadu
            'PY': [r'^PY\d{2}[A-Z]{1,2}\d{4}$'],  # Pondicherry
            'DL': [r'^DL\d{1,2}[A-Z]{1,2}\d{4}$'],  # Delhi
            'KA': [r'^KA\d{2}[A-Z]{1,2}\d{4}$'],  # Karnataka
            'HP': [r'^HP\d{2}[A-Z]\d{4}$'],  # Himachal Pradesh
            'AP': [r'^AP\d{2}[A-Z]{1,2}\d{4}$'],  # Andhra Pradesh
            'MH': [r'^MH\d{2}[A-Z]{1,2}\d{4}$'],  # Maharashtra
            'TS': [r'^TS\d{2}[A-Z]{1,2}\d{4}$'],  # Telangana
            'GJ': [r'^GJ\d{2}[A-Z]{1,2}\d{4}$']   # Gujarat
        }
        
        state_code = text[:2]
        if state_code in patterns:
            return any(bool(re.match(pattern, text)) for pattern in patterns[state_code])
        return False

    def process_plate_text(self, text):
        cleaned_text = ''.join(e for e in text if e.isalnum()).upper()
        
        state_formats = {
            'KL': {'district_len': 2, 'series_len': 2},
            'TN': {'district_len': 2, 'series_len': 2},
            'PY': {'district_len': 2, 'series_len': 2},
            'DL': {'district_len': 2, 'series_len': 2},
            'KA': {'district_len': 2, 'series_len': 2},
            'HP': {'district_len': 2, 'series_len': 1},
            'AP': {'district_len': 2, 'series_len': 2},
            'MH': {'district_len': 2, 'series_len': 2},
            'TS': {'district_len': 2, 'series_len': 2},
            'GJ': {'district_len': 2, 'series_len': 2}
        }
        
        state_code = cleaned_text[:2]
        if state_code in state_formats:
            if self.validate_state_plates(cleaned_text):
                return cleaned_text
            else:
                if len(cleaned_text) >= 8:
                    format_info = state_formats[state_code]
                    state = cleaned_text[:2]
                    district = cleaned_text[2:2+format_info['district_len']]
                    series = cleaned_text[2+format_info['district_len']:
                                        2+format_info['district_len']+format_info['series_len']]
                    number = cleaned_text[-4:]
                    return f"{state}{district}{series}{number}"
        
        return cleaned_text

    def detect_vehicle_type(self, frame):
        results = self.vehicle_model(frame)
        for result in results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            if confidence > 0.7:
                if class_id in [2, 3, 4]:  # motorcycle, scooter, bicycle
                    x1, y1, x2, y2 = map(int, result[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(frame, "Two Wheeler", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    return True
                elif class_id in [5, 7]:  # car, bus, truck
                    x1, y1, x2, y2 = map(int, result[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Car/Vehicle", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return False

    def check_helmet_violations(self, frame):
        results = self.helmet_model(frame)
        num_passengers = 0
        helmet_violations = 0
        high_confidence_violation = False
        
        for result in results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            if confidence > 0.8:  # Increased confidence threshold
                x1, y1, x2, y2 = map(int, result[:4])
                if class_id == 1:  # Helmet
                    num_passengers += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    helmet_violations += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "No Helmet!", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    high_confidence_violation = True
        
        if num_passengers > 2 and max([r.conf[0] for r in results[0].boxes]) > 0.8:
            cv2.putText(frame, "Triple Riding Detected!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return "Helmet violation - Triple riding", True
        elif helmet_violations > 0 and high_confidence_violation:
            return f"Helmet violation - {helmet_violations} person(s) without helmet", True
        return "No violation", False

    def check_seatbelt_violations(self, frame):
        results = self.seatbelt_model(frame)
        seatbelt_violations = 0
        persons_detected = 0
        
        for result in results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            
            if confidence > 0.6:
                x1, y1, x2, y2 = map(int, result[:4])
                
                if class_id == 0:  # Person
                    persons_detected += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                elif class_id == 1:  # Seatbelt
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Seatbelt On", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if persons_detected > 0 and persons_detected > len(results[0].boxes):
            seatbelt_violations = persons_detected - len(results[0].boxes)
            cv2.putText(frame, f"Seatbelt Violation: {seatbelt_violations} person(s)", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return f"Seatbelt violation - {seatbelt_violations} person(s) without seatbelt", True
        
        return "No seatbelt violation", False


    def get_vehicle_bbox(self, frame, plate_bbox):
        """Get the vehicle bounding box associated with the detected plate"""
        x1_plate, y1_plate, x2_plate, y2_plate = plate_bbox
        results = self.vehicle_model(frame)
        
        for result in results[0].boxes.data:
            x1_veh, y1_veh, x2_veh, y2_veh = map(int, result[:4])
            class_id = int(result[5])
            confidence = float(result[4])
            
            # Check if plate is within or near this vehicle bbox
            if (x1_veh <= x1_plate <= x2_veh and 
                y1_veh <= y1_plate <= y2_veh):
                return (x1_veh, y1_veh, x2_veh, y2_veh, class_id, confidence)
        return None

    def process_frame(self, frame):
        current_time = time.time()
        plate_results = self.plate_model(frame)
        
        # Check for violations on vehicles
        results = self.vehicle_model(frame)
        for result in results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            
            x1_veh, y1_veh, x2_veh, y2_veh = map(int, result[:4])
            vehicle_roi = frame[y1_veh:y2_veh, x1_veh:x2_veh]
            
            if confidence > 0.8:
                violation = "No violation"
                is_violation = False
                
                if class_id in [2, 3, 4]:  # Two-wheelers
                    violation, is_violation = self.check_helmet_violations(vehicle_roi)
                else:  # Cars and other vehicles
                    front_seat_roi = vehicle_roi[:int(vehicle_roi.shape[0]/2), :]
                    violation, is_violation = self.check_seatbelt_violations(front_seat_roi)
                
                if not is_violation:
                    violation = "No violation"
                
                # Only save unknown models if there's a violation
                if is_violation:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = f'violation_frames/frame_UNKNOWN_{timestamp}.jpg'
                    cv2.imwrite(img_path, frame)
                    self.save_detection("UNKNOWN", "Unknown Model", violation, img_path)
                    cv2.putText(frame, f"{violation}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Process vehicles with detected plates
        for result in plate_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                
                if conf > 0.25:
                    vehicle_info = self.get_vehicle_bbox(frame, (int(x1), int(y1), int(x2), int(y2)))
                    if not vehicle_info:
                        continue
                    
                    x1_veh, y1_veh, x2_veh, y2_veh, class_id, confidence = vehicle_info
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size == 0:
                        continue
                    
                    enhanced_roi = self.preprocess_for_ocr(roi)
                    results = self.reader.readtext(enhanced_roi)
                    
                    if results:
                        text = ' '.join([result[1] for result in results])
                        cleaned_text = self.process_plate_text(text)
                        
                        for state in self.state_codes:
                            if cleaned_text.startswith(state):
                                final_plate = cleaned_text
                                
                                if len(final_plate) >= 8:
                                    if final_plate not in self.processed_plates or \
                                    current_time - self.processed_plates[final_plate] > 60:
                                        
                                        self.processed_plates[final_plate] = current_time
                                        model_name = self.fetch_vehicle_model(final_plate)
                                        
                                        # Only proceed if model is known
                                        if model_name != 'Unknown Model':
                                            is_bike = class_id in [2, 3, 4]
                                            vehicle_roi = frame[y1_veh:y2_veh, x1_veh:x2_veh]
                                            
                                            violation = "No violation"
                                            is_violation = False
                                            
                                            if is_bike:
                                                violation, is_violation = self.check_helmet_violations(vehicle_roi)
                                            else:
                                                front_seat_roi = vehicle_roi[:int(vehicle_roi.shape[0]/2), :]
                                                violation, is_violation = self.check_seatbelt_violations(front_seat_roi)
                                            
                                            if not is_violation:
                                                violation = "No violation"
                                            
                                            save_folder = 'violation_frames' if is_violation else 'detected_vehicles'
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            img_path = f'{save_folder}/frame_{final_plate}_{timestamp}.jpg'
                                            cv2.imwrite(img_path, frame)
                                            
                                            if is_violation:
                                                cv2.putText(frame, f"{violation}", (10, 90),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                            
                                            self.save_detection(final_plate, model_name, violation, img_path)
                                            self.draw_info(frame, int(x1), int(y1), int(x2-x1), int(y2-y1), 
                                                        final_plate, model_name, violation)
        
        return frame







    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.putText(processed_frame, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Vehicle Detection System', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = VehicleDetectionSystem()
    system.run(r'c:\Users\jacob\OneDrive\Desktop\project needs\testimages\tripletest.mp4')
