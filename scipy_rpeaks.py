import pandas as pd
import numpy as np
from scipy.signal import find_peaks, detrend
import scipy.stats

global_col = None

def get_mad(vals):
  median = np.median(vals)
  return np.median(np.abs(vals-median))

# col has no NaN
def scipy_rpeaks(col, sampling_rate, args):
  # Detrending (Baseline Wander Removal)
  global_col = col
  window_size = 500
  df_clean = pd.DataFrame()
  df_clean['Original'] = pd.Series(col)
  # df_clean['Baseline'] = df_clean['Original'].rolling(window=window_size, center=True, min_periods=1).mean()
  # df_clean['ECG_detrended'] = df_clean['Original'] - df_clean['Baseline']
  df_clean['ECG_detrended'] = detrend(df_clean['Original'])

  # --- 3. Determine Polarity and Invert if necessary ---
  # max_val = np.max(df_clean['ECG_detrended'])
  # min_val = np.min(df_clean['ECG_detrended'])
  # if np.abs(min_val) > np.abs(max_val) * 1.5: # 1.5 is a margin to confirm inversion

  signal_skew = scipy.stats.skew(df_clean['ECG_detrended'])
  if signal_skew < 0:
      df_clean["ECG_Clean"] = df_clean["ECG_detrended"] * -1
      inversion_status = "Inverted"
  else:
      df_clean["ECG_Clean"] = df_clean["ECG_detrended"] 
      inversion_status = "Standard"

  # --- 4. Find R-peaks ---
  if args.prominence:
    prominence_threshold = args.prominence
  else:
    # Recalculate robust prominence threshold on the final (potentially inverted) signal.
    # Use 40% of the range (99% - 1%) as a slightly stricter threshold to reduce false positives.
    # prominence_threshold = robust_range * 0.45
    robust_range = np.percentile(df_clean["ECG_Clean"], 99) - np.percentile(df_clean["ECG_Clean"], 1)

    # the R-peak is usually significantly higher than $6\sigma$
    prominence_threshold = max(6 * get_mad(df_clean["ECG_Clean"]), robust_range * 0.45)

  # Find peaks on the final signal, only looking for positive deflections (R-peaks)
  peaks_indices, properties = find_peaks(
      df_clean["ECG_Clean"],
      prominence=prominence_threshold,
      distance=args.distance,
      height=args.height,
      threshold=args.threshold,
      width=args.width,
      wlen=args.wlen,
      rel_height=args.rel_height,
      plateau_size=args.plateau_size,
  )

  return df_clean["ECG_Clean"], peaks_indices
