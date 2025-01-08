import cv2
import easyocr
import os
import re
from datetime import datetime
from ultralytics import YOLO

class VehicleDetectionSystem:
    def __init__(self):
        os.makedirs('detected_vehicles_plates', exist_ok=True)
        self.text_file = "detected_plates.txt"
        self.initialize_text_file()

        # Initialize EasyOCR and YOLO models
        self.reader = easyocr.Reader(['en'])
        self.custom_model = YOLO(r'models/newcombinedmodel.pt')

        # Define plate patterns for validation
        self.state_plate_patterns = {
            'KL': [r'^KL\d{2}[A-Z]{2}\d{4}$', r'^KL\d{2}[A-Z]\d{4}$'],
            'TN': [r'^TN\d{2}[A-Z]{1,2}\d{4}$'],
            'AP': [r'^AP\d{2}[A-Z]{1,2}\d{4}$'],
            'MH': [r'^MH\d{2}[A-Z]{1,2}\d{4}$'],
            'KA': [r'^KA\d{2}[A-Z]{1,2}\d{4}$'],
            'TS': [r'^TS\d{2}[A-Z]{1,2}\d{4}$'],
            'DL': [r'^DL\d{1,2}[A-Z]{1,2}\d{4}$'],
            'GJ': [r'^GJ\d{2}[A-Z]{1,2}\d{4}$'],
            'HP': [r'^HP\d{2}[A-Z]\d{4}$'],
        }
    def validate_state_plates(self, text):
        """
        Validates the state code at the start of the license plate.
        """
        import re
        patterns = {
            'KL': [r'^KL\d{2}[A-Z]{2}\d{4}$', r'^KL\d{2}[A-Z]\d{4}$'],
            'TN': [r'^TN\d{2}[A-Z]{1,2}\d{4}$'],
            'AP': [r'^AP\d{2}[A-Z]{1,2}\d{4}$'],
            'MH': [r'^MH\d{2}[A-Z]{1,2}\d{4}$'],
            'KA': [r'^KA\d{2}[A-Z]{1,2}\d{4}$'],
            'TS': [r'^TS\d{2}[A-Z]{1,2}\d{4}$'],
            'DL': [r'^DL\d{1,2}[A-Z]{1,2}\d{4}$'],
            'GJ': [r'^GJ\d{2}[A-Z]{1,2}\d{4}$'],
            'HP': [r'^HP\d{2}[A-Z]\d{4}$']
        }
        
        # Get state code from the text
        state_code = text[:2]
        if state_code in patterns:
            # Check against patterns
            return any(bool(re.match(pattern, text)) for pattern in patterns[state_code])
        return False

    def initialize_text_file(self):
        """Initialize the text file to store valid detected plates."""
        if not os.path.exists(self.text_file):
            with open(self.text_file, 'w') as f:
                f.write("Timestamp,Plate Text\n")

    def validate_plate_format(self, plate_text):
        """Validate the plate text against predefined patterns."""
        state_code = plate_text[:2]
        if state_code in self.state_plate_patterns:
            for pattern in self.state_plate_patterns[state_code]:
                if re.match(pattern, plate_text):
                    return True
        return False

    def save_detected_plate(self, plate_text):
        """Save valid detected plate text with a timestamp to the text file."""
        if self.validate_plate_format(plate_text):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.text_file, 'a') as f:
                f.write(f"{timestamp},{plate_text}\n")
            print(f"Valid Plate Detected and Saved: {plate_text}")
        else:
            print(f"Invalid Plate Skipped: {plate_text}")

    def preprocess_for_ocr(self, roi):
        """Preprocess the ROI for better OCR results."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    def process_frame(self, frame):
        results = self.custom_model(frame)
        for result in results[0].boxes.data:
            class_id = int(result[5])
            confidence = float(result[4])
            x1, y1, x2, y2 = map(int, result[:4])

            if class_id == 2 and confidence > 0.2:  # License plate detection
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Preprocess and OCR
                processed_roi = self.preprocess_for_ocr(roi)
                ocr_results = self.reader.readtext(processed_roi)

                if ocr_results:
                    # Combine and clean text
                    plate_text = ' '.join([result[1] for result in ocr_results]).strip()
                    plate_text = ''.join(e for e in plate_text if e.isalnum()).upper()

                    if plate_text and self.validate_state_plates(plate_text):
                        print(f"Detected Plate: {plate_text}")
                        self.save_detected_plate(plate_text)


    def run(self, image_path):
        """Run the system on an image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error loading image")
            return

        # Resize while maintaining aspect ratio
        target_height = 720
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        frame = cv2.resize(frame, (new_width, target_height))

        # Process the frame
        self.process_frame(frame)
        cv2.imshow('Vehicle Detection System', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = VehicleDetectionSystem()
    system.run(r'c:\Users\jacob\OneDrive\Desktop\project needs\testimages\sensors-23-04335-g011-550.jpg')
