# Vehicle Violation Detection System

This project is a Vehicle Violation Detection System that identifies traffic violations such as missing helmets and triple riding on motorcycles, using computer vision techniques. It includes two main components:

you can download my custom trained yolo models from: https://drive.google.com/drive/folders/1T7j03oBzUlBEAtS2aVXxLMTmpygHzoDK?usp=sharing

1. **Local GUI (`gui.py`)**: A desktop application built with Tkinter for viewing and managing violation data stored in Google Drive, designed to run locally on your machine.
2. **Colab Processing Script (`collabcode.py`)**: A script for processing video files in Google Colab, leveraging YOLO models and EasyOCR to detect vehicles, license plates, and violations, saving results to Google Drive.

## Features
- Detects motorcycle-related violations (e.g., no helmet, triple riding).
- Extracts and validates Indian license plate numbers using OCR.
- Stores violation data and images in Google Drive.
- Provides a GUI to filter, view, and manually update violation records.

## Prerequisites

### General Requirements
- Python 3.8 or higher (for local GUI).
- A Google account with access to Google Drive and Google Colab (for Colab script).
- Basic familiarity with running Python scripts and using Google Colab.

### Local GUI (`gui.py`)
- A local machine (Windows, macOS, or Linux).
- Google Drive API credentials (see setup below).

### Colab Script (`collabcode.py`)
- A Google Colab account.
- Pre-trained YOLO models (`best.pt` for custom detection, `yolov8n.pt` for motorcycle detection).
- Access to a GPU in Colab (optional but recommended for faster processing).

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/jacobjohnxn/vehicle-ai-violation-detection.git
cd vehicle-ai-violation-detection
```
### 2. Local GUI (gui.py) Setup
Install Dependencies
Install the required Python packages:

```bash
pip install tk ttkthemes tkcalendar Pillow google-auth-oauthlib google-auth-httplib2 google-api-python-client requests beautifulsoup4

```
### 3.Set Up Google Drive API
1 Go to the Google Cloud Console.
2 Create a new project (e.g., "VehicleViolationSystem").
3 Enable the Google Drive API:
        Navigate to "APIs & Services" > "Library".
        Search for "Google Drive API" and enable it.
4 Configure OAuth Consent Screen:
        Go to "APIs & Services" > "OAuth consent screen".
        Set the app type to "External" and fill in the required details.
5 Create OAuth 2.0 Credentials:
    Go to "APIs & Services" > "Credentials".
    Click "Create Credentials" > "OAuth 2.0 Client IDs".
    Select "Desktop app" as the application type.
    Download the JSON file (e.g., client_secret_*.json) and save it to your project directory (e.g., c:\Users\<your-username>\Downloads\).
6 Update the CREDENTIALS_FILE path in gui.py:
```bash
CREDENTIALS_FILE = r'<path-to-your-json-file>'
```
Replace <path-to-your-json-file> with the actual path to your downloaded JSON file.

### 4. Configure Google Drive Folder
Create a folder in Google Drive (e.g., vehicle_detection1) to store CSV data and images.
Get the folder ID from the URL (e.g., https://drive.google.com/drive/folders/your id → ID: your id).
Update the folder_id in gui.py:

```bash
self.folder_id = "1BlXj8DLgxJ-1IN5FXLfeAzdr9u5MKLbv"  # Replace with your folder ID
```
Create a subfolder named videos (e.g., ID: ) and update the videos_folder_id in the upload_video method if needed.
Run the GUI
```bash
python gui.py
```
The first time you run it, it will prompt you to authenticate with Google Drive via a browser window. Follow the instructions to grant access, and a token.json file will be created in your project directory.
### 5. Colab Script (collabcode.py) Setup
Create a Google Colab Account
Sign up or log in to Google Colab.
Create a new notebook.
Install Dependencies in Colab
Paste and run the following cell at the top of your notebook:

```bash
!pip install easyocr ultralytics opencv-python-headless
```
Upload Models to Google Drive
Download the required YOLO models:
best.pt: Your custom-trained YOLO model for helmet/no-helmet/license plate detection.
yolov8n.pt: Pre-trained YOLOv8n model (download from Ultralytics YOLOv8).
Upload both models to your Google Drive folder (e.g., /MyDrive/vehicle_detection1/).
Configure Google Drive
Ensure your Google Drive folder structure matches:
```bash
/MyDrive/vehicle_detection1/
  ├── best.pt
  ├── yolov8n.pt
  ├── detected_vehicles/  (auto-created)
  ├── violation_frames/   (auto-created)
  ├── processed_videos/   (auto-created)
  ├── vehicle_data.csv    (auto-created)
  └── videos/             (create this manually)
```
Upload video files (e.g., .mp4, .avi) to the videos subfolder.
Run the Colab Script
Copy the entire collabcode.py content into a Colab cell.
Run the cell. It will:
Mount your Google Drive.
Process all videos in the videos folder.
Save results (CSV and images) to the respective subfolders.
Move processed videos to processed_videos.
Enable GPU (Optional)
For faster processing, enable GPU in Colab:
Go to "Runtime" > "Change runtime type" > Select "GPU" > Save.
Usage
Local GUI (gui.py)
Launch the GUI with python gui.py.
Use the interface to:
Filter violations by date, vehicle type, violation type, or search by plate number/vehicle type.
View violation details and images.
Manually enter license plates for "Unknown" detections.
Upload new videos to Google Drive for processing.
Refresh data from Google Drive.
Colab Script (collabcode.py)
Upload videos to /MyDrive/vehicle_detection1/videos/.
Run the script in Colab to process videos and save results.
Check /MyDrive/vehicle_detection1/vehicle_data.csv for violation logs and subfolders for images.
Project Structure
```bash

vehicle-ai-violation-detection/
├── gui.py              # Local GUI script
├── collabcode.py       # Colab processing script
├── README.md           # This file
└── <your-json-file>.json  # Google Drive API credentials (not tracked in Git)
```
Notes
Ensure your Google Drive has sufficient storage for videos, images, and CSV data.
The Colab script assumes videos are in the videos folder; adjust videos_dir in collabcode.py if needed.
For local GUI, keep the token.json file secure and do not share it publicly.
The system supports Indian license plate formats; modify state_codes and validate_state_plates in collabcode.py for other regions.
Troubleshooting
GUI Authentication Fails: Check the CREDENTIALS_FILE path and ensure the JSON file is valid.
Colab GPU Not Working: Verify GPU runtime is enabled and reinstall dependencies.
No Violations Detected: Ensure best.pt is trained correctly and video quality is sufficient.
Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the project!

License
This project is licensed under the MIT License. See  for details.

