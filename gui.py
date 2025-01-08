import tkinter as tk
from tkinter import ttk
import csv
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
from tkcalendar import DateEntry

class VehicleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection Viewer")
        self.root.geometry("1200x800")
        
        # Read CSV data
        self.all_vehicle_data = {}
        self.vehicle_data = {}
        self.current_image_index = 0
        with open('vehicle_data.csv', 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                model = row['Vehicle Model']
                if model not in self.all_vehicle_data:
                    self.all_vehicle_data[model] = []
                self.all_vehicle_data[model].append(row)
        
        self.vehicle_data = self.all_vehicle_data.copy()

        # Create main panels
        left_panel = ttk.Frame(root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        right_panel = ttk.Frame(root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Date Filter
        date_frame = ttk.Frame(left_panel)
        date_frame.pack(pady=10)
        ttk.Label(date_frame, text="Select Date:").pack()
        self.date_picker = DateEntry(date_frame, width=12, background='darkblue',
                                   foreground='white', borderwidth=2)
        self.date_picker.pack(pady=5)
        ttk.Button(date_frame, text="Apply Date Filter", 
                  command=self.apply_date_filter).pack(pady=5)
        ttk.Button(date_frame, text="Clear Filter", 
                  command=self.clear_filter).pack(pady=5)

        # Dropdown for vehicle models
        ttk.Label(left_panel, text="Select Vehicle Model:").pack(pady=10)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(left_panel, textvariable=self.model_var)
        self.model_dropdown['values'] = list(self.vehicle_data.keys())
        self.model_dropdown.pack(pady=5)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.update_display)

        # Navigation buttons
        nav_frame = ttk.Frame(left_panel)
        nav_frame.pack(pady=10)
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)

        # Image display
        self.image_label = ttk.Label(right_panel)
        self.image_label.pack()

        # Details display
        self.details_text = tk.Text(right_panel, height=10, width=50)
        self.details_text.pack(pady=10)

        # Counter label
        self.counter_label = ttk.Label(right_panel, text="")
        self.counter_label.pack()

    def apply_date_filter(self):
        selected_date = self.date_picker.get_date().strftime("%Y%m%d")
        self.vehicle_data = {}
        
        for model, detections in self.all_vehicle_data.items():
            filtered_detections = []
            for detection in detections:
                detection_date = detection['Timestamp'][:8]  # Get YYYYMMDD part
                if detection_date == selected_date:
                    filtered_detections.append(detection)
            if filtered_detections:
                self.vehicle_data[model] = filtered_detections
        
        # Update dropdown values
        self.model_dropdown['values'] = list(self.vehicle_data.keys())
        self.current_image_index = 0
        self.model_var.set('')  # Reset selection
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, f"Filtered for date: {selected_date}")

    def clear_filter(self):
        self.vehicle_data = self.all_vehicle_data.copy()
        self.model_dropdown['values'] = list(self.vehicle_data.keys())
        self.current_image_index = 0
        self.model_var.set('')
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, "Filter cleared")

    def update_display(self, event=None):
        self.current_image_index = 0
        self.show_current_detection()

    def show_current_detection(self):
        selected_model = self.model_var.get()
        if not selected_model or selected_model not in self.vehicle_data:
            return

        detections = self.vehicle_data[selected_model]
        if not detections:
            return

        self.counter_label.config(text=f"Image {self.current_image_index + 1} of {len(detections)}")

        detection = detections[self.current_image_index]
        self.details_text.delete(1.0, tk.END)
        details = f"Timestamp: {detection['Timestamp']}\n"
        details += f"Plate Number: {detection['Plate Number']}\n"
        details += f"Violation: {detection['Violation']}\n"
        details += f"Image Path: {detection['Image Path']}\n"
        self.details_text.insert(tk.END, details)

        if os.path.exists(detection['Image Path']):
            img = cv2.imread(detection['Image Path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (800, 600))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img)
            self.image_label.image = img

    def next_image(self):
        selected_model = self.model_var.get()
        if selected_model:
            max_index = len(self.vehicle_data[selected_model]) - 1
            self.current_image_index = min(self.current_image_index + 1, max_index)
            self.show_current_detection()

    def prev_image(self):
        self.current_image_index = max(0, self.current_image_index - 1)
        self.show_current_detection()

def main():
    root = tk.Tk()
    app = VehicleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
