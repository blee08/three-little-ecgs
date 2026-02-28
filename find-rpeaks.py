# Load NeuroKit and other useful packages
# /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/neurokit2
# made some changes in 
import argparse
import os
import neurokit2 as nk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
# https://github.com/AlexandreDecan/portion
import portion as P
from collections import defaultdict
import scipy_rpeaks
import pprint

from typing import List, Tuple, Dict, Any

# uses qt5agg for interactive display.
matplotlib.use('qt5agg')
cols_index = (1, 2, 3)
cols_name = ('B', 'C', 'D')
colors = ["blue", "green", "orange"]
sampling_rate = 500
# good_methods = ["scipy", "kalidas2017", "pantompkins", "manikandan2012"]
good_methods = ["scipy"]
good_methods = ["scipy", "kalidas2017", "pantompkins"]
# good_methods = ["scipy"]

# error_rows=500 * 600 * 2# starting to x after this. 119.5 minutes.

# error_rows=3585000 # starting to x after this. 119.5 minutes.
# error_rows=3570000 # starting to x after this. 119 minutes.
# error_rows=3570000 # starting to x after this. 119 minutes.
# error_rows=3000000 # starting to x 100 minutes
# error_rows=9000000 + 500 * 60 * 50 # starting to x 350 minutes
# error_rows=500 * 60 * 60 # 60 minutes
# error_rows=0
# skip_rows=11 + error_rows
# max_rows=500*60*60 # 60 minutes
# max_rows=500*60*30 # 30 minutes
# max_rows=500*60*60 # 60 minutes

def safe_float_converter(val):
    try:
        return float(val)
    except ValueError:
        return 0.0 # Return 0.0 for bad values

experiment_id="pig24-D82"
fname = "./../../../pig-data/" + experiment_id + "-file-1.csv"

experiment_id="pig21-D2"
fname = "./../../../pig-data/" + experiment_id + ".csv"


# manikandan2012 didn't work for this
experiment_id="20240322_P21_2-06_00_00_000_11_59_59_998.ascii"
fname = "./../../../pig-data/" + experiment_id

# manikandan2012 didn't work for this
experiment_id="20240322_P21_1-00_00_00_000_05_59_59_998.ascii"
fname = "./../../../pig-data/" + experiment_id


def get_col_name(index):
  return "col" + cols_name[index]

def get_mad(vals):
  median = np.median(vals)
  return np.median(np.abs(vals-median))

def calculate_robust_snr(signal):
  # Use 99th percentile to represent the R-peak signal level
  signal_level = np.percentile(np.abs(signal), 99)

  # Use Median Absolute Deviation (MAD) for a robust noise floor
  # MAD is better than std() because it isn't inflated by the R-peaks
  noise_level = get_mad(signal)

  if noise_level == 0:
      return 0

  return signal_level / noise_level

def filter_rpeaks(signals, rpeaks, sampling_rate=sampling_rate):
  # use 0.1 second padding before and after
  pad = sampling_rate // 10;
  maxindex = len(signals)
  adjusted_signals = [np.max(signals[max(0, index-pad):min(index+pad, maxindex)]) 
                      for index in rpeaks] 
  adjusted_signals = adjusted_signals - np.min(adjusted_signals)
  mad = get_mad(adjusted_signals)
  threshold = np.median(adjusted_signals) - mad/0.6745 * 3.5
  filtered_rpeaks = [rpeaks[i] for i in range(len(rpeaks)) if adjusted_signals[i] >= threshold] 
  print("maganitude threshold: " + str(threshold) + " median: " +
        str(np.median(adjusted_signals)) + " mad: " + str(mad))
    
  return filtered_rpeaks

def find_gaps(rpeaks):
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
          gaps.append(P.openclosed(rpeaks[i], rpeaks[i+1]))
    return gaps


def merge_intervals(intervals: List[P]) -> List[P]:
    """
    Merges overlapping intervals and returns the total non-overlapping duration in seconds.
    The input intervals are assumed to be sorted by start time.
    e.g., [(10, 30), (25, 40)] -> merged to [(10, 40)] -> duration = 30 seconds
    
    Args:
        sorted intervals which may be overlapping.
    
    Returns:
        List of sorted non-overlapping intervals
    """
    if not intervals:
        return 0

    merged = []
    tupled_intervals = P.to_data(intervals)
    _, current_start, current_end, _ = tupled_intervals[0]

    for _, next_start, next_end, _ in tupled_intervals[1:]:
        if next_start < current_end:
            # Overlap or touch, merge the intervals
            current_end = max(current_end, next_end)
        else:
            # No overlap, finish the current merged interval and start a new one
            merged.append(P.closedopen(current_start, current_end))
            current_start, current_end = next_start, next_end

    # Add the last merged interval
    merged.append(P.closedopen(current_start, current_end))
    return merged

warmup_interval = 30 * sampling_rate
interval_size = 1800 * sampling_rate 
skip_description = 11
# skip_description = 11 + 1800 * sampling_rate

num_cols = len(cols_index)

# TODO: fix all_data, currently getting the last iteration data only

# return true if we get to the end of the file.
def process_rpeaks(index, args, all_data, draw=False):
  starting_point = index * interval_size
  warmup_end = starting_point + warmup_interval
  skiprows = skip_description + starting_point
  max_rows = interval_size + warmup_interval
  print("skiprows: " + str(skiprows) + " max_rows: " + str(max_rows))
  fname = os.path.join(args.input_dir, args.inputfile)
  print(fname)

  cols = np.loadtxt(fname, delimiter=",", usecols=cols_index, skiprows=skiprows,
                    max_rows=max_rows, converters=safe_float_converter)

  num_seconds_after_warmup = (len(cols) - warmup_interval) / sampling_rate
  print("num_seconds: " + str(num_seconds_after_warmup))
  print("\nindex = " + str(index))

  for method in args.methods:
    print("\tmethod = " + method)
    combined_signals = pd.DataFrame()
    combined_rpeaks = []
    combined_gaps = []
    combined_bpm = []
    for i in range(0, len(cols_index)):
      signal = info = None # placeholder
      if method == "scipy" or method == "scipy2":
        clean_signal, rpeaks = scipy_rpeaks.scipy_rpeaks(cols[:,i],
                                                         sampling_rate, args)
      else:
        # Automatically process the (raw) ECG signal
        s, b = nk.ecg_invert(cols[:,i], sampling_rate=sampling_rate)
        signals, info = nk.ecg_process(s, sampling_rate=sampling_rate, method=method)
        combined_signals.insert(i, get_col_name(i), signals["ECG_Clean"])
        clean_signal = signals["ECG_Clean"]
        rpeaks = info["ECG_R_Peaks"]

      # filtered_rpeaks = filter_rpeaks(signals["ECG_Clean"], info["ECG_R_Peaks"])
      filtered_rpeaks = filter_rpeaks(clean_signal, rpeaks)
      print("i: " + str(i) + " num_rpeaks: " + str(len(filtered_rpeaks)))

      filtered_rpeaks = [p + starting_point for p in filtered_rpeaks if p >=
                         warmup_interval]
      all_data["rpeaks"][method][i] += filtered_rpeaks
      all_data["snr"][method][i].append(calculate_robust_snr(clean_signal))

      # removing gaps in rpeaks
      gaps = find_gaps(filtered_rpeaks)
      gaps_after = merge_intervals([P.closedopen(starting_point, warmup_end)] + gaps)
      # adjust the first gap because it's due to the warmup period
      if gaps_after[0].upper == warmup_end:
        gaps_after = gaps_after[1:]
      else:
        gaps_after = [P.closedopen(warmup_end, gaps_after[0].upper)] + gaps_after[1:]

      print("max gap in min: " + str(gaps_after[-1].upper/sampling_rate/60 if gaps_after else 0))

      # TODO: should move this to post-process
      total_gap = sum([gap.upper-gap.lower for gap in gaps]) / sampling_rate
      total_gap_after = sum([gap.upper-gap.lower for gap in gaps_after]) / sampling_rate 

      num_rpeaks_after = len([index for index in filtered_rpeaks if index >= warmup_end])
      original_num_rpeaks_after = len([index for index in rpeaks if index >= warmup_interval])
      hr_filter = num_rpeaks_after/(num_seconds_after_warmup) *60;
      hr_filter_gap = (num_rpeaks_after - len(gaps_after)) / (num_seconds_after_warmup - total_gap_after)*60

      print("#rpeaks (filtered): (all, after_30s): (" +
            str(len(filtered_rpeaks)) + ", " + str(num_rpeaks_after) + ")")
      print("#rpeaks (original): (all, after_30s): (" + str(len(rpeaks)) + ", " + str(original_num_rpeaks_after) + ")")
      print("#rpeaks (del gap): (all, after_30s): (" +
            str(len(filtered_rpeaks) - len(gaps)) + ", " 
            + str(num_rpeaks_after - len(gaps_after)) + ")")
      print("total_gap in sec: " + str(total_gap) + ", total_gap_after: " +
            str(total_gap_after))
      print("heart-rates (after 30s): filtered, filter-del-gap: " + 
            f'{hr_filter:.2f}' + " " + f'{hr_filter_gap:.2f}')

      # combined_rpeaks += [info["ECG_R_Peaks"]]
      combined_rpeaks += [filtered_rpeaks]
      combined_gaps += [gaps_after]
      combined_bpm += [hr_filter_gap]

    # pprint.pprint(all_data["snr"][method])

  all_data["total_sec"] += num_seconds_after_warmup

  # similar = compare_three_values(combined_hrs)
  # print("similarity:")
  # print(similar)

  if draw:
    # nk.ecg_plot(signals, info);
    # plt.title(method)
    # plt.xlim(0, 500*60)
    # plt.savefig("plots/ecg-details-col" + cols_name[i] + method+ ".png")
    # plt.show(block=False)

    plot = nk.events_plot(combined_rpeaks, combined_signals,
                          scale_factor=1.0/sampling_rate/60)
    plt.title(method)
    # plt.xlim(500*60 * 19, 500*60*21)
    xmin_index = sampling_rate * 60 * 0
    xmax_index = sampling_rate * 60 * 2
    plt.xlim(xmin_index, xmax_index)
    for i in range(len(cols_index)):
      for gap in combined_gaps[i]:
        plt.hlines(y=-0.425 - 0.05*i, xmin=gap.lower, xmax=gap.upper,
                   color=colors[i], linewidth=2)
    plot.show()
    # plt.savefig("./../plots/combined-" + method + "-119min-121min-gap-detection.png", dpi=600)
    # plt.savefig("./../plots/combined-" + method + "-119min-121min-gap-detection.pdf", dpi=600)

  # Return whether we went to the end of the file, we are done
  return len(cols) < max_rows


# method="rodrigues"  # ecg_clean() does not like this.
# cannot handle pig-21

# emrich: processing error for pig21 after 10 minutes

# for method in ("kalidas2017", "nabian2018", "pantompkins"):
# for method in ("rodrigues", "emrich", "manikandan2012", "neurokit2",):
# for method in ("manikandan2012", "neurokit2",):
import argparse
from scipy.signal import find_peaks

def main():
  parser = argparse.ArgumentParser(description="Peak detection script using scipy.signal.find_peaks")

  # File and Directory arguments
  parser.add_argument("--inputfile", type=str, default="copy.csv", help="Path to the input data file")
  parser.add_argument("--input_dir", type=str, default=".", help="Directory for processing")
  parser.add_argument("--outputfile", type=str, default=None, help="Path to save the results")
  parser.add_argument("--methods", type=str, nargs="+", default=good_methods, help="Path to save the results")

  # scipy.signal.find_peaks parameters
  # Note: These are set to None by default as per scipy's signature
  # Using nargs='+' allows you to pass two numbers in the command line (e.g., --height 10 50), which Python will treat as a list.
  #
  # Troubleshooting Tip
  # If you are getting "too many" peaks (false positives), the issue is likely
  # that your prominence_threshold is too low. If you are getting "no" peaks,
  # your distance might be too large or your height (if not None) is too high.
  parser.add_argument("--height", type=float, default=None, help="Required height of peaks.")
  parser.add_argument("--threshold", type=float, default=None, help="Required threshold of peaks")
  parser.add_argument("--distance", type=float, default=sampling_rate * 0.35, help="Required minimal horizontal distance between neighbouring peaks")
  parser.add_argument("--prominence", type=float, default=None, help="Required prominence of peaks. If unspecified, uses a quarter of signal range")
  parser.add_argument("--width", type=float, default=None, help="Required width of peaks")
  parser.add_argument("--wlen", type=int, default=None, help="Used for calculating the prominence")
  parser.add_argument("--rel_height", type=float, default=0.5, help="Relative height at which the peak width is measured")
  parser.add_argument("--plateau_size", type=float, default=None, help="Size of flat tops (plateaus) of peaks")

  args = parser.parse_args()
  pprint.pprint(args)

  # Example of how you would call find_peaks with these args:
  # (Assuming 'data' is your signal array)
  # peaks, properties = find_peaks(
  #     data,
  #     height=args.height,
  #     threshold=args.threshold,
  #     distance=args.distance,
  #     prominence=args.prominence,
  #     width=args.width,
  #     wlen=args.wlen,
  #     rel_height=args.rel_height,
  #     plateau_size=args.plateau_size
  # )


  print("args.inputfile: ", args.inputfile)
  print(f"Arguments parsed successfully. Input: {args.inputfile}")

  all_data = {"total_sec": 0, 
            "rpeaks": {m: [[] for _ in range(num_cols)] for m in args.methods},
            "snr": {m: [[] for _ in range(num_cols)] for m in args.methods},
            "snr_window": interval_size // sampling_rate, 
            "inputfile": args.inputfile}

  num_iter = 14
  for index in range(num_iter):
    done = process_rpeaks(index, args, all_data, draw=False)
    if done:
      break

  if True:
    num_minutes = (interval_size / sampling_rate ) // 60
    print("num_minutes: " + str(num_minutes))
    if args.outputfile:
      fname = args.outputfile
    else:
      fname = args.inputfile + ".all_data.npz"

    np.savez(fname, **all_data)
    #"all_data_" + experiment_id + "_" + str(num_minutes) + "x" + str(num_iter) + "scipy.npz"

  # fig = plt.gcf() fig.set_size_inches(10, 12, forward=True) fig.savefig(“myfig.png”)
  #nk.ecg_plot(signals, info);
  #plt.show(block=False)

if __name__ == "__main__":
  main()

