
# Load NeuroKit and other useful packages
# /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/neurokit2
# made some changes in 
import neurokit2 as nk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import portion as P

# uses qt5agg for interactive display.
matplotlib.use('qt5agg')
cols_index = (1, 2, 3)
cols_name = ('B', 'C', 'D')
colors = ["blue", "green", "orange"]
sampling_rate = 500
# fname = "../../pig-data/pig24-D82-file-1-shorter.csv"

# TODO: There are colums with 'x' data. E.g., line 3585423.
# Come up with a way to gracefully handle this.

# error_rows=500 * 600 * 2# starting to x after this. 119.5 minutes.
error_rows=0
error_rows=3585000 # starting to x after this. 119.5 minutes.
error_rows=3570000 # starting to x after this. 119 minutes.
error_rows=3570000 # starting to x after this. 119 minutes.
error_rows=3000000 # starting to x 100 minutes
skip_rows=11 + error_rows
max_rows=500*60*60 # 60 minutes
max_rows=500*60*1 # 1 minute
max_rows=500*60*3 # 3 minute
max_rows=500*60*10 # 10 minutes
max_rows=500*60*5 # 3 minute
max_rows=500*60*30 # 30 minutes

num_seconds = max_rows / 500

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
  # stdev = np.std(adjusted_signals)
  # print("mad: " + str(mad) + " stdev: " + str(stdev), " stdev/mad: " + str(stdev/mad))
  # median_half = np.percentile(adjusted_signals, 50) / 2
  # filtered_rpeaks = rpeaks
  # print("median_half = " + str(median_half) + " median - mad/0.67 * 3: " 
  #      + str(np.median(adjusted_signals) - mad/0.6745*3.5))
  threshold = np.median(adjusted_signals) - mad/0.6745 * 3.5
  filtered_rpeaks = [rpeaks[i] for i in range(len(rpeaks)) if adjusted_signals[i] >= threshold] # done with r-peak filtering\
    
  return filtered_rpeaks


def detect_gaps(rpeaks):
    # detects gaps of bad data to be omitted
    # returns array of intervals for outliers (start/end times)
    intervals = np.diff(rpeaks)
    mad = get_mad(intervals)
    median =  np.median(intervals)
    threshold = median + mad/0.6745 * 3.5
    print("gap threshold: " + str(threshold) + " median: " + str(median) + " mad: " + str(mad))
    outliers = intervals > threshold
    gaps = []
    for i in range(len(outliers)):
        if outliers[i]: 
            gaps.append(P.closed(rpeaks[i], rpeaks[i+1]))
    return gaps

all_signals = {}
all_rpeaks = {}
all_gaps = {}
def process_rpeaks(method, draw=False):
  combined_signals = pd.DataFrame()
  combined_rpeaks = []
  combined_gaps = []
  for i in range(0, len(cols_index)):
    # Automatically process the (raw) ECG signal
    s, b = nk.ecg_invert(cols[:,i], sampling_rate=500)
    # signals, info = nk.ecg_process(s, sampling_rate=500)
    signals, info = nk.ecg_process(s, sampling_rate=500, method=method)
    combined_signals.insert(i, get_col_name(i), signals["ECG_Clean"])

    # uncomment if we want to try filtering logic
    filtered_rpeaks = filter_rpeaks(signals["ECG_Clean"], info["ECG_R_Peaks"])

    # removing gaps in rpeaks
    gaps = detect_gaps(filtered_rpeaks)
    gaps_after = [P.closed(gap.lower if gap.lower >= 30*500 else 30*500,
                           gap.upper) for gap in gaps if gap.upper >= 30 * 500]
    total_gap = sum([gap.upper-gap.lower for gap in gaps]) / sampling_rate
    total_gap_after = sum([gap.upper-gap.lower for gap in gaps_after]) / sampling_rate
    combined_gaps += [gaps]

    num_rpeaks_after = len([index for index in filtered_rpeaks if index >= 30 * 500])
    original_num_rpeaks_after = len([index for index in info["ECG_R_Peaks"] if
                                     index >= 30 * sampling_rate])
    hr_filter = num_rpeaks_after/(num_seconds-30) *60;
    hr_filter_gap = (num_rpeaks_after - len(gaps_after)) / (num_seconds - 30 - total_gap_after)*60

    print("#rpeaks (filtered): (all, after_30s): (" +
          str(len(filtered_rpeaks)) + ", " + str(num_rpeaks_after) + ")")
    print("#rpeaks (original): (all, after_30s): (" + str(len(info["ECG_R_Peaks"])) + ", " 
          + str(original_num_rpeaks_after) + ")")
    print("#rpeaks (del gap): (all, after_30s): (" +
          str(len(filtered_rpeaks) - len(gaps)) + ", " 
          + str(num_rpeaks_after - len(gaps_after)) + ")")
    print("total_gap: " + str(total_gap) + ", total_gap_after: " +
          str(total_gap_after))
    print("heart-rates (after 30s): filtered, filter-del-gap: " + 
          f'{hr_filter:.2f}' + " " + f'{hr_filter_gap:.2f}')

    # combined_rpeaks += [info["ECG_R_Peaks"]]
    combined_rpeaks += [filtered_rpeaks]

  all_signals[method] = combined_signals
  all_rpeaks[method] = combined_rpeaks
  all_gaps[method] = combined_gaps

  if draw:
    # nk.ecg_plot(signals, info);
    # plt.title(method)
    # plt.xlim(0, 500*60)
    # plt.savefig("plots/ecg-details-col" + cols_name[i] + method+ ".png")
    # plt.show(block=False)

    plot = nk.events_plot(combined_rpeaks, combined_signals, scale_factor=1.0/500/60)
    plt.title(method)
    # plt.xlim(500*60 * 19, 500*60*21)
    xmin_index = 500 * 60 * 0
    xmax_index = 500 * 60 * 2
    plt.xlim(xmin_index, xmax_index)
    for i in range(len(cols_index)):
      for gap in combined_gaps[i]:
        plt.hlines(y=-0.425 - 0.05*i, xmin=gap.lower, xmax=gap.upper,
                   color=colors[i], linewidth=2)
    plot.show()
    # plt.savefig("./../plots/combined-" + method + "-119min-121min-gap-detection.png", dpi=600)
    # plt.savefig("./../plots/combined-" + method + "-119min-121min-gap-detection.pdf", dpi=600)

# method="rodrigues"  # ecg_clean() does not like this.
method="emrich"
method="pantompkins"
method="emrich"
method="neurokit2"
method="pantompkins"
method="emrich"
method="pantompkins"

fname = "./../../../pig-data/pig24-D82-file-1.csv"
cols = np.loadtxt(fname, delimiter=",", usecols=cols_index, skiprows=skip_rows,
                  max_rows=max_rows, converters=safe_float_converter)


# for method in ("emrich", "pantompkins", "neurokit2"):
for method in ("emrich", "pantompkins"):
  print("\nmethod = " + method)
  process_rpeaks(method, draw=True)


# fig = plt.gcf() fig.set_size_inches(10, 12, forward=True) fig.savefig(“myfig.png”)
#nk.ecg_plot(signals, info);
#plt.show(block=False)
