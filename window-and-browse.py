import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import mne
import pprint

def processEdf(filename):
  # 2. Load the raw EDF file
  # Setting preload=True loads the data into RAM, which is required for modifications like filtering
  raw = mne.io.read_raw_edf(filename, preload=True)

  # 3. Inspect the channels to locate your ECG channel names
  print("Channel names in file:", raw.ch_names)

  # 4. Explicitly map your channel to 'ecg' type 
  # Replace 'ECG' with the exact name string found in raw.ch_names above
  # mapping = {'ECG': 'ecg'}
  mapping = {'Abdomen_1': 'ecg'}
  raw.set_channel_types(mapping)

  # 5. Extract fundamental info about the data
  info = raw.info
  sampling_rate = info['sfreq']
  print(f"Sampling Rate: {sampling_rate} Hz")
  print(f"Assigned Channel Types: {raw.get_channel_types()}")

  # 6. Extract raw data matrix and times (Optional)
  # This gets data from the 'ecg' channel specifically
  ecg_data, times = raw.copy().pick('ecg').get_data(return_times=True)

  pprint.pprint(ecg_data)

  # 7. Plot the ECG signal
  # You can interactively scroll through your data using MNE's native browser
  # raw.plot(duration=10, n_channels=1, scalings='auto')

  ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, tmin=-0.2, tmax=0.2)

  # Plot average heartbeat
  ecg_epochs.average().plot()

def select_and_read_file():
    # 1. Open the file browser dialog
    file_path = filedialog.askopenfilename(
        title="Select a file to read",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    # 2. If the user picked a file, proceed to read it
    if file_path:
        # Update the label to show the file name
        file_label.config(text=f"File: {os.path.basename(file_path)}", fg="black")
        
        # Clear any text currently inside the display window
        text_display.delete("1.0", tk.END)
        
        try:
          processEdf(file_path)
          #  with open(file_path, 'r', encoding='utf-8') as file:
          #      content = file.read()
          #      # Insert the file contents into the text box
          #      text_display.insert(tk.END, content)
        except UnicodeDecodeError:
            messagebox.showerror("Error", "Could not read this file. Make sure it's a plain text file (UTF-8).")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

# --- GUI Setup ---

# Initialize the main window
root = tk.Tk()
root.title("Text File Reader")
root.geometry("600x450")
root.minsize(400, 300) # Prevents making the window too small to use

# Create a top frame to organize the layout horizontally
top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X, padx=15, pady=15)

# Add the "Browse" button
# Clicking this executes the 'select_and_read_file' function above
browse_btn = tk.Button(
    top_frame, 
    text="Browse File", 
    command=select_and_read_file, 
    padx=10, 
    pady=5
)
browse_btn.pack(side=tk.LEFT)

# Add a label next to the button to show which file is open
file_label = tk.Label(top_frame, text="No file selected", fg="gray", font=("Arial", 10, "italic"))
file_label.pack(side=tk.LEFT, padx=15)

# Add a large, scrollable text area to display the actual content
text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
text_display.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

# Keep the window open and responsive
root.mainloop()
