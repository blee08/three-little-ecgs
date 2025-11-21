import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Setup and Load Data ---
file_name = 'copy.csv'
skip_rows = 11
sampling_rate = 500  # samples per second
analysis_seconds = 10 # Use a small window for the ML demonstration
num_rows_to_read = analysis_seconds * sampling_rate

df = pd.read_csv(
    file_name, skiprows=skip_rows, nrows=num_rows_to_read,
    header=None, usecols=[0, 1, 2, 3], skipinitialspace=True
)
ecg_cols_idx = [1, 2, 3]

# Data Preprocessing (Averaging, Detrending, and Inversion)
for col_idx in ecg_cols_idx:
    df.loc[:, col_idx] = pd.to_numeric(df[col_idx], errors='coerce')

df['ECG_avg'] = df[ecg_cols_idx].mean(axis=1)
df_clean = df.dropna(subset=['ECG_avg']).copy()
df_clean['Time_s'] = df_clean.index / sampling_rate

window_size = 500
df_clean['Baseline'] = df_clean['ECG_avg'].rolling(window=window_size, center=True, min_periods=1).mean()
df_clean['ECG_detrended'] = df_clean['ECG_avg'] - df_clean['Baseline']

# Inversion Check and Correction (based on previous finding)
ecg_signal_final = df_clean['ECG_detrended'].values
if np.abs(np.min(ecg_signal_final)) > np.abs(np.max(ecg_signal_final)) * 1.5:
    ecg_signal_final *= -1
df_clean['ECG_final'] = ecg_signal_final

# --- 2. Initial Lenient Peak Detection (Candidate Peaks) ---
# Find ALL potential peaks (R, P, T, noise) using very low prominence.
# The distance is kept high to avoid detecting multiple points per wave.
candidate_peaks, properties = find_peaks(
    df_clean['ECG_final'],
    prominence=0.01,  # Low threshold to capture everything
    distance=40      # Relatively small distance (80ms)
)

print(f"Total candidate peaks found: {len(candidate_peaks)}")

# --- 3. Feature Extraction ---
def extract_features(signal, peak_indices, sampling_rate):
    features = []
    
    # Calculate RR-intervals
    peak_diffs = np.diff(peak_indices, prepend=peak_indices[0])
    
    for i, idx in enumerate(peak_indices):
        # Feature 1: Amplitude
        amplitude = signal[idx]
        
        # Feature 2: Prominence (re-calculated in the context of all candidates)
        # Using a fixed width for consistent prominence calculation
        prominence, _, _ = find_peaks(signal, indices=[idx], width=20)
        prominence = prominence[0] if len(prominence) > 0 else 0
        
        # Feature 3: Preceding RR Interval (in seconds)
        rr_interval_prev_s = peak_diffs[i] / sampling_rate
        
        # Feature 4: Succeeding RR Interval (in seconds) - use the next diff or 0 if last
        rr_interval_next_s = peak_diffs[i+1] / sampling_rate if i < len(peak_diffs) - 1 else 0
        
        features.append([amplitude, prominence, rr_interval_prev_s, rr_interval_next_s])
        
    return pd.DataFrame(features, columns=['Amplitude', 'Prominence', 'RR_Prev_s', 'RR_Next_s'])

X = extract_features(df_clean['ECG_final'].values, candidate_peaks, sampling_rate)

# --- 4. Simulated Labeling for Training Data ---
# Label based on a reliable, fixed threshold to simulate manual labeling.
# True R-peaks are expected to be the tallest.
max_amp = X['Amplitude'].max()
y = (X['Amplitude'] > max_amp * 0.5).astype(int) # Label as 1 if amplitude > 50% of max

# Check how many R-peaks are simulated for training
print(f"Simulated True R-peaks for training: {y.sum()}")

# --- 5. Random Forest Training and Filtering ---
if y.sum() > 0:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)

    # Predict on the entire candidate set (or test set for validation)
    y_pred_all = rf_classifier.predict(X)

    # Filter the peaks
    final_r_peaks_indices = candidate_peaks[y_pred_all == 1]
    
    # --- 6. Results and Visualization (Optional) ---
    print(f"\nRandom Forest filtered R-peak count: {len(final_r_peaks_indices)}")
    
    # Plotting the result of the ML filtering on the 10-second segment
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 4))
    plt.plot(df_clean['Time_s'], df_clean['ECG_final'], label='Corrected Detrended ECG', color='C1', alpha=0.8)
    plt.plot(
        df_clean.iloc[final_r_peaks_indices]['Time_s'],
        df_clean.iloc[final_r_peaks_indices]['ECG_final'],
        'x',
        label=f'RF Filtered R Peaks ({len(final_r_peaks_indices)})',
        color='green',
        markersize=8,
        linewidth=0
    )
    plt.title('R Peak Detection Filtered by Random Forest Classifier (First 10 Seconds)')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('ECG Amplitude (mV)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('rf_filtered_r_peaks.png')
    print("Plot saved as rf_filtered_r_peaks.png")
else:
    print("\nError: Not enough R-peaks found to create a training set. Please check data or increase analysis window.")
