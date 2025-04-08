# Install required packages
!pip install ultralytics pyyaml roboflow kaggle --quiet

import os
import yaml
from ultralytics import YOLO
import torch
import time
from threading import Thread

def keep_alive():
    while True:
        print("Keeping session alive...")
        time.sleep(1000)  # Every 10 minutes

Thread(target=keep_alive, daemon=True).start()
  # Print every 10 minutes

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")

# Verify your dataset YAML (update path as needed)
data_yaml = '/kaggle/input/march26/dara4/data.yaml'
with open(data_yaml, 'r') as f:
    print(f.read())

# Load YOLOv8x pretrained on COCO
model = YOLO('/kaggle/working/yolov8x.pt')

# Define training parameters
training_params = {
    'data': data_yaml,   # Path to dataset YAML
    'epochs': 100,       # Total epochs
    'imgsz': 640,        # Image size
    'batch': 16,         # Batch size
    'optimizer': 'AdamW',# Optimizer
    'lr0': 0.005,        # Initial learning rate
    'lrf': 0.01,         # Final learning rate factor
    'weight_decay': 0.0005, # Regularization
    'patience': 20,      # Early stopping
    'device': 0,         # Use GPU
    'workers': 4,        # Speed up data loading
    'name': 'yolov8x_custom2', # Experiment name
    'cos_lr': True,      # Cosine learning rate
    'warmup_epochs': 3,  # Short warmup for stability
    'amp': True,         # Mixed precision
    'scale': 0.5,        # Scaling augmentation
    'fliplr': 0.5,       # Horizontal flip
    'mosaic': 0.0,       # Disable mosaic
    'mixup': 0.0,        # Disable mixup
}

# Check for last checkpoint
checkpoint_path = '/kaggle/working/runs/detect/yolov8x_custom22/weights/last.pt'
if os.path.exists(checkpoint_path):
    print(f"Resuming training from {checkpoint_path}...")
    model = YOLO(checkpoint_path)  # Load last saved model
    results = model.train(resume=True)  # Resume training
else:
    print("No checkpoint found. Starting fresh training...")
    results = model.train(**training_params)

# Validate the model
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")

# Export the trained model
