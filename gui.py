import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import csv
import os
import sys
import subprocess
from PIL import Image, ImageTk
from datetime import datetime
from tkcalendar import DateEntry
from ttkthemes import ThemedTk
import tkinter.messagebox as messagebox
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
import logging
import requests
from bs4 import BeautifulSoup
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = r'c:\Users\jacob\Downloads\yoursecrect.json'

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def authenticate_google_drive():
    """Authenticate with Google Drive API."""
    creds = None
    token_path = resource_path('token.json')
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(resource_path(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def get_file_id(service, file_name, folder_id=None):
    """Get the file ID from Google Drive by file name within a specific folder."""
    try:
        query = f"name='{file_name}' and trashed=false"
        if folder_id:
            query = f"'{folder_id}' in parents and {query}"
        
        results = service.files().list(
            q=query,
            fields="files(id, modifiedTime)"
        ).execute()

        files = results.get('files', [])
        if files:
            file_id = files[0]['id']
            modified_time = files[0]['modifiedTime']
            logger.info(f"Found {file_name} with ID {file_id}, last modified at {modified_time}")
            return file_id
        else:
            logger.info(f"No non-trashed {file_name} found{' in folder ' + folder_id if folder_id else ''}")
            return None
    except Exception as e:
        logger.error(f"Error getting file ID for {file_name}{' in folder ' + folder_id if folder_id else ''}: {e}")
        return None


def download_file(service, file_id, destination):
    """Download a file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return destination
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return None

def upload_file(service, file_path, file_name, folder_id, file_id=None):
    """Upload a file to Google Drive, updating if file_id exists, creating in folder_id if not."""
    try:
        media = MediaFileUpload(file_path, mimetype='text/csv')
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
            logger.info(f"Updated {file_name} with ID {file_id}")
        else:
            metadata = {'name': file_name, 'parents': [folder_id]}
            service.files().create(body=metadata, media_body=media).execute()
            logger.info(f"Created new {file_name} in folder {folder_id}")
    except Exception as e:
        logger.error(f"Failed to upload {file_name} to Google Drive: {e}")

def remove_file_with_retry(file_path, max_attempts=5, delay=1):
    """Remove a file with retry mechanism for permission errors."""
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                logger.error(f"Failed to remove {file_path} after {max_attempts} attempts")



class AutocompleteEntry(tk.Entry):
    def __init__(self, suggestions, parent_app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suggestions = suggestions
        self.parent_app = parent_app  # Reference to main app to call search function
        self.var = self["textvariable"] = tk.StringVar()
        self.var.trace("w", self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Down>", self.move_down)
        self.bind("<Return>", self.selection)
        self.bind("<Escape>", self.hide_list)
        self.lb_up = False

    def changed(self, name, index, mode):
        search_term = self.var.get().lower()
        if search_term == "":
            self.hide_list()
        else:
            matches = [s for s in self.suggestions if search_term in s.lower()]
            if matches:
                if not self.lb_up:
                    self.lb = tk.Listbox()
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<ButtonRelease-1>", self.single_click_selection)
                    self.lb.bind("<Right>", self.selection)
                    self.lb.bind("<Return>", self.selection)
                    self.lb.place(x=self.winfo_x(), y=self.winfo_y() + self.winfo_height())
                    self.lb_up = True
                self.lb.delete(0, tk.END)
                for match in matches:
                    self.lb.insert(tk.END, match)
            else:
                self.hide_list()

    def single_click_selection(self, event):
        """Handle single click on listbox item"""
        if self.lb_up:
            self.var.set(self.lb.get(tk.ACTIVE))
            self.hide_list()
            self.icursor(tk.END)
            # Trigger search immediately after selection
            self.parent_app.apply_search_filter()
        return "break"

    def selection(self, event):
        if self.lb_up:
            self.var.set(self.lb.get(tk.ACTIVE))
            self.hide_list()
            self.icursor(tk.END)
            # Trigger search immediately after selection
            self.parent_app.apply_search_filter()
        return "break"

    def move_down(self, event):
        if self.lb_up:
            self.lb.focus()
            self.lb.selection_set(0)
        return "break"

    def hide_list(self, *args):
        if self.lb_up:
            self.lb.destroy()
            self.lb_up = False

class IntroPage(tk.Frame):
    def __init__(self, parent, switch_callback):
        super().__init__(parent)
        self.configure(bg='#f0f0f0')
        self.switch_callback = switch_callback
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        content_frame = tk.Frame(self, bg='#f0f0f0')
        content_frame.grid(row=1, column=0, sticky='nsew')
        content_frame.grid_rowconfigure((0, 1, 2), weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        title = ttk.Label(content_frame, 
                         text="Welcome to Vehicle Violation Detection", 
                         font=("Arial", 24, "bold"), 
                         background="#f0f0f0", 
                         foreground="#4CAF50")
        title.grid(row=0, column=0, pady=20, sticky='n')
        
        desc = ttk.Label(content_frame, 
                        text="Monitor and analyze vehicle violations with ease.\n"
                             "View detailed reports and images of detected incidents.",
                        font=("Arial", 14), 
                        background="#f0f0f0", 
                        justify="center")
        desc.grid(row=1, column=0, pady=20, sticky='n')
        
        start_button = ttk.Button(content_frame, 
                               text="Get Started", 
                               command=self.start_app, 
                               style="TButton")
        start_button.grid(row=2, column=0, pady=20, sticky='n')
        
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)

    def start_app(self):
        self.fade_out()

    def fade_out(self, alpha=1.0):
        if alpha > 0:
            self.configure(bg=f'#{int(240 - (1-alpha)*240):02x}{int(240 - (1-alpha)*240):02x}{int(240 - (1-alpha)*240):02x}')
            self.after(50, self.fade_out, alpha - 0.1)
        else:
            self.switch_callback()

class VehicleGUI(tk.Frame):
    def __init__(self, parent, drive_service):
        super().__init__(parent)
        self.configure(bg='#f0f0f0')
        self.drive_service = drive_service
        self.folder_id = "yourfolderid from google drive or implemnt other method if u use other cloud"  # Your folder ID
        self.all_vehicle_data = {'All Violations': []}
        self.vehicle_data = {}
        self.current_image_index = 0
        self.image_cache = {}
        self.plate_cache = {}
        
        self.grid_rowconfigure(2, weight=1)  # Content area expands
        self.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.load_and_update_data()
        self._create_ui()
        self.apply_filters()
        self.bind("<Configure>", self.on_resize)  # Bind resize event

    def _create_ui(self):
        # Title Frame
        title_frame = tk.Frame(self, bg="#4CAF50")
        title_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, 
                                text="Vehicle Violation Detection", 
                                font=("Arial", 18, "bold"), 
                                background="#4CAF50", 
                                foreground="white")
        title_label.pack(expand=True, pady=10)

        search_frame = ttk.LabelFrame(self, text="Search")
        search_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
        for col in range(8):
            search_frame.grid_columnconfigure(col, weight=0)
        
        # Date Filter Section
        ttk.Label(search_frame, text="Select Date:").grid(row=0, column=0, padx=5, sticky='w')
        self.date_picker = DateEntry(search_frame, width=12,
                                    background='#2196F3',
                                    foreground='white',
                                    borderwidth=2)
        self.date_picker.grid(row=0, column=1, padx=5, sticky='w')
        self.date_picker.bind("<<DateEntrySelected>>", lambda e: self.apply_filters())        
        ttk.Button(search_frame, text="Reset Date",
        command=self.reset_date_filter).grid(row=0, column=2, padx=5, sticky='w')

        ttk.Label(search_frame, text="Vehicle Type:").grid(row=0, column=3, padx=5, sticky='w')
        # Get unique vehicle types, handle None/empty values
        unique_vehicle_types = sorted(set(d.get('Vehicle Type', 'Unknown') for d in self.all_vehicle_data['All Violations'] if d.get('Vehicle Type')))
        vehicle_types_list = ["All"] + [vtype for vtype in unique_vehicle_types if vtype]
        self.vehicle_type_combobox = ttk.Combobox(search_frame, values=vehicle_types_list, state="readonly", width=15)
        self.vehicle_type_combobox.set("All")
        self.vehicle_type_combobox.grid(row=0, column=4, padx=5, sticky='w')
        self.vehicle_type_combobox.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        # Violation Type Filter
        ttk.Label(search_frame, text="Violation Type:").grid(row=0, column=5, padx=5, sticky='w')
        # Get unique violation types, handle None/empty values
        unique_violation_types = sorted(set(d.get('Violation', 'Unknown') for d in self.all_vehicle_data['All Violations'] if d.get('Violation')))
        violation_types_list = ["All"] + [vtype for vtype in unique_violation_types if vtype]
        self.violation_type_combobox = ttk.Combobox(search_frame, values=violation_types_list, state="readonly", width=15)
        self.violation_type_combobox.set("All")
        self.violation_type_combobox.grid(row=0, column=6, padx=5, sticky='w')
        self.violation_type_combobox.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        # Refresh Button
        ttk.Button(search_frame, text="↻", width=3,
                command=self.refresh_data).grid(row=0, column=7, padx=5, sticky='w')

        # Search Section
        ttk.Label(search_frame, text="Search:").grid(row=1, column=0, padx=5, sticky='w')
        plate_numbers = [d.get('Plate Number', '') for d in self.all_vehicle_data['All Violations']]
        vehicle_types = [d.get('Vehicle Type', '') for d in self.all_vehicle_data['All Violations']]
        suggestions = sorted(set(plate_numbers + vehicle_types))
        self.search_entry = AutocompleteEntry(suggestions, self, search_frame, width=20)
        self.search_entry.grid(row=1, column=1, columnspan=2, padx=5, sticky='ew')
        self.search_entry.bind("<Return>", lambda event: self.apply_filters())
        # Action Buttons
        # Add your widgets, e.g.:
        ttk.Button(search_frame, text="Realtime", command=self.run_realtime).grid(row=1, column=3, padx=5, sticky='w')
        ttk.Button(search_frame, text="Upload Video", command=self.upload_video).grid(row=1, column=4, padx=5, sticky='w')
        self.manual_entry_button = ttk.Button(search_frame, text="Manual Entry", command=self.manual_entry)
        self.manual_entry_button.grid(row=1, column=5, padx=5, sticky='w')
        self.manual_entry_button.grid_remove()        # Content Area

        content_frame = tk.Frame(self, bg='#f0f0f0')
        content_frame.grid(row=2, column=0, columnspan=3, sticky='nsew', pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)

        # Details Section
        details_frame = tk.Frame(content_frame, bg='#f0f0f0')
        details_frame.grid(row=0, column=0, padx=10, sticky='nsew')
        details_frame.grid_rowconfigure(0, weight=1)
        details_frame.grid_columnconfigure(0, weight=1)

        self.details_text = tk.Text(details_frame,
                                height=15,
                                width=40,
                                font=("Arial", 11))
        self.details_text.grid(row=0, column=0, sticky='nsew')

        scrollbar = ttk.Scrollbar(details_frame,
                                orient="vertical",
                                command=self.details_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.details_text.config(yscrollcommand=scrollbar.set)

        # Image Display
        self.image_label = tk.Label(content_frame, bg='#f0f0f0')
        self.image_label.grid(row=0, column=1, padx=10, sticky='nsew')

        # Navigation Section
        nav_frame = tk.Frame(self, bg='#f0f0f0')
        nav_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky='ew')
        nav_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ttk.Button(nav_frame, text="Previous",
                command=self.prev_image).grid(row=0, column=0, padx=5, sticky='e')
        ttk.Button(nav_frame, text="Next",
                command=self.next_image).grid(row=0, column=1, padx=5, sticky='w')

        self.counter_label = ttk.Label(nav_frame, text="")
        self.counter_label.grid(row=0, column=2, padx=20, sticky='w')

        # Placeholder Image
        self.placeholder_img = ImageTk.PhotoImage(
            Image.new('RGB', (400, 300), color='#d3d3d3'))
        self.image_label.config(image=self.placeholder_img)

        self.update_display()

    def update_combobox_values(self):
        # Update Vehicle Type combobox
        unique_vehicle_types = sorted(set(d.get('Vehicle Type', 'Unknown') for d in self.all_vehicle_data['All Violations'] if d.get('Vehicle Type')))
        vehicle_types_list = ["All"] + [vtype for vtype in unique_vehicle_types if vtype]
        self.vehicle_type_combobox['values'] = vehicle_types_list
        
        # Update Violation Type combobox
        unique_violation_types = sorted(set(d.get('Violation', 'Unknown') for d in self.all_vehicle_data['All Violations'] if d.get('Violation')))
        violation_types_list = ["All"] + [vtype for vtype in unique_violation_types if vtype]
        self.violation_type_combobox['values'] = violation_types_list

    def refresh_data(self):
        self.load_and_update_data()
        self.update_combobox_values()  # Add this line
        self.update_display()
        # Update suggestions for search
        plate_numbers = [d.get('Plate Number', '') for d in self.all_vehicle_data['All Violations']]
        vehicle_types = [d.get('Vehicle Type', '') for d in self.all_vehicle_data['All Violations']]
        self.search_entry.suggestions = sorted(set(plate_numbers + vehicle_types))

    def on_resize(self, event):
        """Handle window resize to adjust image size."""
        self.show_current_detection()

    def load_and_update_data(self):
        try:
            # Reset data
            self.all_vehicle_data = {'All Violations': []}
            
            # Search for the file within the specific folder
            results = self.drive_service.files().list(
                q=f"'{self.folder_id}' in parents and name='vehicle_data.csv' and trashed=false",
                fields="files(id)"
            ).execute()
            
            files = results.get('files', [])
            file_id = files[0]['id'] if files else None
            logger.info(f"Loading data - File ID: {file_id if file_id else 'None'}")

            if file_id:
                temp_csv = resource_path('temp_vehicle_data.csv')
                downloaded_file = download_file(self.drive_service, file_id, temp_csv)
                if downloaded_file:
                    logger.info(f"Successfully downloaded vehicle_data.csv to {temp_csv}")
                    with open(temp_csv, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if 'Vehicle Type' not in row or not row['Vehicle Type']:
                                plate_number = row.get('Plate Number', 'Unknown')
                                vehicle_type = self.fetch_vehicle_model(plate_number)
                                row['Vehicle Type'] = vehicle_type
                            self.all_vehicle_data['All Violations'].append(row)
                    self.save_data_to_drive(temp_csv)
                    remove_file_with_retry(temp_csv)
                else:
                    logger.error("Download failed; starting with empty data")
                    self.all_vehicle_data = {'All Violations': []}
            else:
                logger.info("No file found; starting with empty data")

            self.vehicle_data = self.all_vehicle_data.copy()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.vehicle_data = {'All Violations': []}

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
            if response.status_code != 200:
                return 'Unknown Model 200'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            model_tag = soup.select_one("p.input_vehical_layout_vehicalModel__1ABTF")
            if model_tag:
                model_name = model_tag.text.strip()
                self.plate_cache[plate_number] = model_name
                return model_name
            return 'Unknown Model 300'
        except Exception as e:
            return 'Unknown Model 400'

    def save_data_to_drive(self, temp_csv):
        try:
            with open(temp_csv, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['Timestamp', 'Plate Number', 'Violation', 'Image Path', 'Vehicle Type']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for detection in self.all_vehicle_data['All Violations']:
                    writer.writerow(detection)
            
            file_id = get_file_id(self.drive_service, "vehicle_data.csv", self.folder_id)
            upload_file(self.drive_service, temp_csv, "vehicle_data.csv", self.folder_id, file_id)
        except Exception as e:
            logger.error(f"Failed to save and upload CSV: {e}")

    def apply_date_filter(self):
        selected_date = self.date_picker.get_date().strftime("%Y%m%d")
        filtered_data = [
            d for d in self.all_vehicle_data['All Violations']
            if d['Timestamp'].startswith(selected_date)
        ]
        self.vehicle_data = {'All Violations': filtered_data}
        self.current_image_index = 0
        self.update_display()

    def apply_vehicle_type_filter(self):
        selected_type = self.vehicle_type_combobox.get()
        if selected_type == "All":
            self.vehicle_data = self.all_vehicle_data.copy()
        else:
            filtered_data = [
                d for d in self.all_vehicle_data['All Violations']
                if d.get('Vehicle Type', 'Unknown') == selected_type
            ]
            self.vehicle_data = {'All Violations': filtered_data}
        self.current_image_index = 0
        self.update_display()

    def apply_search_filter(self):
        search_text = self.search_entry.get().strip().lower()
        if not search_text:
            self.vehicle_data = self.all_vehicle_data.copy()
        else:
            filtered_data = [
                d for d in self.all_vehicle_data['All Violations']
                if search_text in d.get('Plate Number', '').lower() or search_text in d.get('Vehicle Type', '').lower()
            ]
            self.vehicle_data = {'All Violations': filtered_data}
        self.current_image_index = 0
        self.update_display()

    def reset_date_filter(self):
        self.vehicle_data = self.all_vehicle_data.copy()
        self.current_image_index = 0
        self.update_display()

    def apply_violation_filter(self):
        selected_violation = self.violation_type_combobox.get()
        if selected_violation == "All":
            self.vehicle_data = self.all_vehicle_data.copy()
        else:
            filtered_data = [
                d for d in self.all_vehicle_data['All Violations']
                if d.get('Violation', 'Unknown') == selected_violation
            ]
            self.vehicle_data = {'All Violations': filtered_data}
        self.current_image_index = 0
        self.update_display()

    def update_display(self):
        violations = self.vehicle_data['All Violations']
        if not violations:
            self.clear_display()
            self.details_text.insert(1.0, "No violations found for selected filter")
            return
        self.show_current_detection()

    def show_current_detection(self):
        violations = self.vehicle_data['All Violations']
        if not violations:
            return

        detection = violations[self.current_image_index]
        self.counter_label.config(
            text=f"Violation {self.current_image_index + 1} of {len(violations)}")

        self.details_text.delete(1.0, tk.END)
        details = (
            f"Time: {detection['Timestamp']}\n"
            f"Plate Number: {detection.get('Plate Number', 'Unknown')}\n"
            f"Violation: {detection.get('Violation', 'Not specified')}\n"
            f"Vehicle Type: {detection.get('Vehicle Type', 'Unknown')}\n"
        )
        self.details_text.insert(1.0, details)

    # Show or hide the Manual Entry button based on Plate Number
        plate_number = detection.get('Plate Number', '').lower()
        if plate_number == 'unknown':
            self.manual_entry_button.grid()
        else:
            self.manual_entry_button.grid_remove()

        image_path = detection.get('Image Path', '')
        if image_path:
            self.load_and_display_image(image_path)
        else:
            self.image_label.config(image=self.placeholder_img)


    def load_and_display_image(self, image_path):
        def load_image_async():
            try:
                filename = os.path.basename(image_path)
                file_id = get_file_id(self.drive_service, filename)
                if file_id :
                    temp_image = resource_path(f'temp_image_{self.current_image_index}.jpg')
                    download_file(self.drive_service, file_id, temp_image)
                    img = Image.open(temp_image)
                    width, height = self.image_label.winfo_width(), self.image_label.winfo_height()
                    if width > 1 and height > 1:
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    else:
                        img = img.resize((400, 300), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.image_label.config(image=photo)
                    self.image_label.image = photo
                    remove_file_with_retry(temp_image)
                else:
                    self.image_label.config(image=self.placeholder_img)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                self.image_label.config(image=self.placeholder_img)
        
        threading.Thread(target=load_image_async, daemon=True).start()

    def clear_display(self):
        self.details_text.delete(1.0, tk.END)
        self.image_label.config(image=self.placeholder_img)
        self.counter_label.config(text="")

    def next_image(self):
        violations = self.vehicle_data['All Violations']
        if violations and self.current_image_index < len(violations) - 1:
            self.current_image_index += 1
            self.show_current_detection()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_detection()

    def run_realtime(self):
        script_path = resource_path('local/realtimetemp.py')
        if os.path.exists(script_path):
            try:
                subprocess.Popen([sys.executable, script_path])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run pythonapp.py: {e}")
        else:
            messagebox.showerror("Error", "pythonapp.py not found")

    def upload_video(self):
        file_path = tk.filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            try:
                progress_dialog = UploadProgressDialog(self)
                file_name = os.path.basename(file_path)
                
                # Use the specific videos folder ID
                videos_folder_id = "your video folder id"
                
                progress_dialog.update_progress(20, "Preparing upload...")
                
                media = MediaFileUpload(
                    file_path, 
                    mimetype='video/*',
                    resumable=True,
                    chunksize=1024*1024
                )
                
                file_metadata = {
                    'name': file_name,
                    'parents': [videos_folder_id]
                }
                
                request = self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                )
                
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        progress_dialog.update_progress(
                            progress, 
                            f"Uploading: {progress}%"
                        )
                
                progress_dialog.update_progress(100, "Upload complete!")
                self.after(1000, progress_dialog.destroy)
                messagebox.showinfo("Success", "Video uploaded successfully!")
                
            except Exception as e:
                if 'progress_dialog' in locals():
                    progress_dialog.destroy()
                messagebox.showerror("Error", f"Failed to upload video: {str(e)}")

    def refresh_data(self):
        messagebox.showinfo("Refreshing Data", "Data is being refreshed...")

        self.load_and_update_data()
        self.update_combobox_values()
        self.apply_filters()

        # Update search suggestions
        plate_numbers = [d.get('Plate Number', '') for d in self.all_vehicle_data['All Violations']]
        vehicle_types = [d.get('Vehicle Type', '') for d in self.all_vehicle_data['All Violations']]
        self.search_entry.suggestions = sorted(set(plate_numbers + vehicle_types))  

        messagebox.showinfo("Success", "Data has been refreshed successfully!")

    def get_filtered_data(self):
        data = self.all_vehicle_data['All Violations']
        
        # Apply date filter
        selected_date = self.date_picker.get_date().strftime("%Y%m%d")
        if selected_date:
            data = [d for d in data if d['Timestamp'].startswith(selected_date)]
        
        # Apply vehicle type filter
        selected_type = self.vehicle_type_combobox.get()
        if selected_type != "All":
            data = [d for d in data if d.get('Vehicle Type', 'Unknown') == selected_type]
        
        # Apply violation type filter
        selected_violation = self.violation_type_combobox.get()
        if selected_violation != "All":
            data = [d for d in data if d.get('Violation', 'Unknown') == selected_violation]
        
        # Apply search filter
        search_text = self.search_entry.get().strip().lower()
        if search_text:
            data = [d for d in data if search_text in d.get('Plate Number', '').lower() or 
                    search_text in d.get('Vehicle Type', '').lower()]
        
        return data

    def apply_filters(self):
        self.vehicle_data['All Violations'] = self.get_filtered_data()
        self.current_image_index = 0
        self.update_display()

    def manual_entry(self):
        """Allow manual entry of license plate for entries with 'Unknown' plate number."""
        violations = self.vehicle_data['All Violations']
        if not violations or self.current_image_index >= len(violations):
            messagebox.showinfo("Info", "No violation selected or available.")
            return

        current_detection = violations[self.current_image_index]
        plate_number = current_detection.get('Plate Number', '').lower()

        if plate_number == 'unknown':
            dialog = ManualEntryDialog(self)
            self.wait_window(dialog)
            
            if dialog.result:
                new_plate = dialog.result.strip().upper()
                if new_plate:
                    # Update the current detection
                    current_detection['Plate Number'] = new_plate
                    # Optionally fetch vehicle type
                    new_vehicle_type = self.fetch_vehicle_model(new_plate)
                    current_detection['Vehicle Type'] = new_vehicle_type
                    
                    # Update the all_vehicle_data
                    for i, d in enumerate(self.all_vehicle_data['All Violations']):
                        if d['Timestamp'] == current_detection['Timestamp'] and d.get('Image Path') == current_detection.get('Image Path'):
                            self.all_vehicle_data['All Violations'][i] = current_detection.copy()
                    
                    # Save and upload updated data
                    temp_csv = resource_path('temp_vehicle_data.csv')
                    self.save_data_to_drive(temp_csv)
                    remove_file_with_retry(temp_csv)
                    
                    # Refresh the display and comboboxes
                    self.update_combobox_values()
                    self.apply_filters()
                    messagebox.showinfo("Success", f"Updated Plate Number to {new_plate} and synced with Google Drive.")
        else:
            messagebox.showinfo("Info", "Manual entry is only available for 'Unknown' plate numbers.")

class ManualEntryDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Manual License Plate Entry")
        self.geometry("300x150")
        self.resizable(False, False)
        self.result = None
        
        # Center the dialog
        self.transient(parent)
        self.grab_set()
        
        # Label and Entry
        ttk.Label(self, text="Enter License Plate Number:").pack(pady=10)
        self.entry = ttk.Entry(self, width=20)
        self.entry.pack(pady=5)
        self.entry.focus_set()
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Submit", command=self.submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def submit(self):
        self.result = self.entry.get()
        self.destroy()
    
    def cancel(self):
        self.result = None
        self.destroy()

class UploadProgressDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Uploading Video")
        self.geometry("300x150")
        self.resizable(False, False)
        
        # Center the dialog
        self.transient(parent)
        self.grab_set()
        
        # Progress label
        self.label = ttk.Label(self, text="Uploading: 0%")
        self.label.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self, 
            orient="horizontal",
            length=200, 
            mode="determinate"
        )
        self.progress.pack(pady=10)
        
    def update_progress(self, value, text=None):
        self.progress["value"] = value
        if text:
            self.label["text"] = text
        self.update_idletasks()



class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Violation Detection System")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)  # Minimum window size
        
        self.container = tk.Frame(self.root)
        self.container.pack(expand=True, fill='both')
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.drive_service = authenticate_google_drive()
        
        self.intro_page = IntroPage(self.container, self.show_main_app)
        self.intro_page.grid(row=0, column=0, sticky='nsew')
        
        self.main_app = VehicleGUI(self.container, self.drive_service)

    def show_main_app(self):
        self.intro_page.grid_forget()
        self.main_app.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

    def on_closing(self):
        self.main_app.save_data_to_drive(resource_path('temp_vehicle_data.csv'))
        self.root.destroy()

def main():
    root = ThemedTk(theme="arc")
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()