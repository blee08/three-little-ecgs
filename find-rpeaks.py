
# Load NeuroKit and other useful packages
# /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/neurokit2
# made some changes in 
import neurokit2 as nk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# uses qt5agg for interactive display.
matplotlib.use('qt5agg')
cols_index = (1, 2, 3)
cols_name = ('B', 'C', 'D')
# fname = "../../pig-data/pig24-D82-file-1-shorter.csv"
fname = "../../pig-data/pig24-D82-file-1.csv"

# TODO: There are colums with 'x' data. E.g., line 3585423.
# Come up with a way to gracefully handle this.

# error_rows=500 * 600 * 2# starting to x after this. 119.5 minutes.
error_rows=0
error_rows=3585000 # starting to x after this. 119.5 minutes.
error_rows=3000000 # starting to x 100 minutes
skip_rows=11 + error_rows
max_rows=500*60*60 # 60 minutes
# max_rows=500*60*5 # 5 minutes
max_rows=500*60*30 # 30 minutes
max_rows=500*60*60 # 60 minutes
max_rows=500*60*1 # 1 minute
max_rows=500*60*3 # 3 minute
max_rows=500*60*10 # 10 minutes
max_rows=500*60*30 # 30 minutes
max_rows=500*60*3 # 3 minute
max_rows=500*60*10 # 10 minutes
max_rows=500*60*1 # 1 minute
max_rows=500*60*30 # 30 minutes
max_rows=500*60*3 # 3 minute
max_rows=500*60*30 # 30 minutes

def safe_float_converter(val):
    try:
        return float(val)
    except ValueError:
        return 0.0 # Return 0.0 for bad values

def get_col_name(index):
  return "col" + cols_name[index]

def get_mad(vals):
  median = np.median(vals)
  return np.median(np.abs(vals-median))

# 11/15/2025: not really useful for emrich or pan tompkins for now
# may be useful to detect windows of low-quality data
def filter_rpeaks(signals, rpeaks, sampling_rate=500):
  # use 0.1 second padding before and after
  pad = sampling_rate // 10;
  adjusted_signals = [np.max(signals[index-pad:index+pad]) for index in
                           rpeaks]
  adjusted_signals = adjusted_signals - np.min(adjusted_signals)
  mad = get_mad(adjusted_signals)
  stdev = np.std(adjusted_signals)
  print("mad: " + str(mad) + " stdev: " + str(stdev), " stdev/mad: " + str(stdev/mad))
  median_half = np.percentile(adjusted_signals, 50) / 2
  # filtered_rpeaks = rpeaks
  print("median_half = " + str(median_half) + " median - mad/0.67 * 3: " 
        + str(np.median(adjusted_signals) - mad/0.6745*3.5))
  threshold = np.median(adjusted_signals) - mad/0.6745 * 3.5
  filtered_rpeaks = [rpeaks[i] for i in range(len(rpeaks)) if adjusted_signals[i] >= threshold]
  return filtered_rpeaks

def detect_gaps(rpeaks):
    # detects gaps of bad data to be omitted
    # returns 2D array of times of outliers (start and end)
    intervals = np.diff(rpeaks)
    mad = get_mad(intervals)
    median =  np.median(intervals)
    threshold = median + mad/0.6745 * 3.5
    outliers = intervals > threshold
    array = []
    for i in range(1, len(outliers)):
        if outliers[i]: 
            array += [[rpeaks[i-1], rpeaks[i]]
    return array

            



def process_rpeaks(method, draw=False):
  combined_signals = pd.DataFrame()
  combined_rpeaks = []
  for i in range(0, len(cols_index)):
    # Automatically process the (raw) ECG signal
    s, b = nk.ecg_invert(cols[:,i], sampling_rate=500)
    # signals, info = nk.ecg_process(s, sampling_rate=500)
    signals, info = nk.ecg_process(s, sampling_rate=500, method=method)
    combined_signals.insert(i, get_col_name(i), signals["ECG_Clean"])

    # uncomment if we want to try filtering logic
    filtered_rpeaks = filter_rpeaks(signals["ECG_Clean"], info["ECG_R_Peaks"])

    num_rpeaks_after = len([index for index in filtered_rpeaks if index >= 30 * 500])
    original_num_rpeaks_after = len([index for index in info["ECG_R_Peaks"] if index >= 30 * 500])

    print("#rpeaks (filtered): (all, after_30s): (" +
          str(len(filtered_rpeaks)) + ", " + str(num_rpeaks_after) + ")")
    print("#rpeaks (original): (all, after_30s): (" + str(len(info["ECG_R_Peaks"])) + ", " 
          + str(original_num_rpeaks_after) + ")")

    combined_rpeaks += [info["ECG_R_Peaks"]]
    # combined_rpeaks += [filtered_rpeaks]

  if draw:
    # nk.ecg_plot(signals, info);
    # plt.title(method)
    # plt.xlim(0, 500*60)
    # plt.savefig("plots/ecg-details-col" + cols_name[i] + method+ ".png")
    # plt.show(block=False)

    plot = nk.events_plot(combined_rpeaks, combined_signals, scale_factor=1.0/500/60)
    plt.title(method)
    plt.xlim(500*60 * 19, 500*60*21)
    plot.show()
    plt.savefig("plots/combined-" + method + "-119min-121min.png", dpi=600)
    plt.savefig("plots/combined-" + method + "-119min-121min.pdf", dpi=600)

# method="rodrigues"  # ecg_clean() does not like this.
method="emrich"
method="pantompkins"
method="emrich"
method="neurokit2"
method="pantompkins"
method="emrich"
method="pantompkins"

cols = np.loadtxt(fname, delimiter=",", usecols=cols_index, skiprows=skip_rows,
                  max_rows=max_rows, converters=safe_float_converter)


for method in ("emrich", "pantompkins", "neurokit2"):
  print("\nmethod = " + method)
  process_rpeaks(method, draw=True)


# fig = plt.gcf() fig.set_size_inches(10, 12, forward=True) fig.savefig(“myfig.png”)
#nk.ecg_plot(signals, info);
#plt.show(block=False)
