import argparse
import datetime
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
# https://github.com/AlexandreDecan/portion
import portion as P
import pprint

from typing import List, Tuple, Dict, Any

# uses qt5agg for interactive display.
matplotlib.use('qt5agg')
cols_index = (1, 2, 3)
cols_name = ('B', 'C', 'D')
colors = ["blue", "green", "orange"]
sampling_rate = 500
# fname = "../../pig-data/pig24-D82-file-1-shorter.csv"

# Heart rate < 30 BPM (RR > 2.0s) or > 300 BPM (RR < 0.2s)
MAX_RR_LIMIT = 2.0 * sampling_rate
MIN_RR_LIMIT = 0.2 * sampling_rate

def get_mad(vals):
  median = np.median(vals)
  return np.median(np.abs(vals-median))

def find_gaps(rpeaks):
    # detects gaps of bad data to be omitted
    # returns array of intervals for outliers (start/end times)
    intervals = np.diff(rpeaks)
    mad = get_mad(intervals)
    median =  np.median(intervals)
    threshold = median + mad/0.6745 * 3.5
    print("gap threshold: " + str(threshold) + " median: " + str(median) + " mad: " + str(mad))
    outliers = intervals > threshold
    physio_outliers = (intervals > MAX_RR_LIMIT) | (intervals < MIN_RR_LIMIT)
    gaps = []
    for i in range(len(intervals)):
        if outliers[i] or physio_outliers[i]: 
          gaps.append(P.openclosed(rpeaks[i], rpeaks[i+1]))
    return gaps

def compare_three_values(values):
  # values is a list of positive numbers
  # returns which values are similar
  v_min, v_mid, v_max = sorted(values)
  percentage = 0.03
  is_max_close = (v_max - v_mid) <= (v_max * percentage)
  is_min_close = (v_mid - v_min) <= (v_mid * percentage)

  if is_max_close and is_min_close:
    return [True] * len(values)
  if not is_max_close and not is_min_close:
    return [False] * len(values)
  if is_max_close:
    # exclude min
    return np.array(values) > v_min
  # min and mid are close, so exclude max
  return np.array(values) < v_max


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

def combine_intervals(list_a: List[P], list_b: List[P]) -> List[P]:
    """
    Merges two separate, sorted lists of intervals into a single list,
    maintaining sort order and consolidating any overlaps in a single O(N_a + N_b) sweep.

    Args:
        list_a: First sorted list of [start, end) tuples.
        list_b: Second sorted list of [start, end) tuples.
        
    Returns:
        A single, sorted, and non-overlapping list of [start, end) tuples.
    """

    # 1. Two-Pointer Merge (Combines two sorted lists into one sorted list)
    ptr_a, ptr_b = 0, 0
    all_intervals = []

    # Loop until one list is exhausted, adding the interval with the earliest start time
    while ptr_a < len(list_a) and ptr_b < len(list_b):
      if list_a[ptr_a].lower <= list_b[ptr_b].lower: 
        all_intervals.append(list_a[ptr_a])
        ptr_a += 1
      else:
        all_intervals.append(list_b[ptr_b])
        ptr_b += 1

    # Append any remaining elements
    all_intervals.extend(list_a[ptr_a:])
    all_intervals.extend(list_b[ptr_b:])

    # 2. Consolidate Overlaps (Since all_intervals is now sorted, we use the helper)
    return merge_intervals(all_intervals)

def combine_points(list_a: List[int], list_b: List[int]) -> List[P]:
    """
    Merges two separate, sorted lists of integer points into a single list,
    maintaining sort order and consolidating any overlaps in a single O(N_a + N_b) sweep.

    Args:
        list_a: First sorted list
        list_b: Second sorted list
        
    Returns:
        A single, sorted list 
    """

    # 1. Two-Pointer Merge (Combines two sorted lists into one sorted list)
    ptr_a, ptr_b = 0, 0
    all_intervals = []

    # Loop until one list is exhausted, adding the interval with the earliest start time
    while ptr_a < len(list_a) and ptr_b < len(list_b):
      if list_a[ptr_a] <= list_b[ptr_b]: 
        all_intervals.append(list_a[ptr_a])
        ptr_a += 1
      else:
        all_intervals.append(list_b[ptr_b])
        ptr_b += 1

    # Append any remaining elements
    all_intervals.extend(list_a[ptr_a:])
    all_intervals.extend(list_b[ptr_b:])
    return all_intervals


# fig = plt.gcf() fig.set_size_inches(10, 12, forward=True) fig.savefig(“myfig.png”)
#nk.ecg_plot(signals, info);
#plt.show(block=False)


# Define the window size (10 minutes = 600 seconds)
WINDOW_SIZE = 300 * sampling_rate

def calculate_time_series_metrics(
    gaps: List[P], event_timestamps: List[int], num_series: int = 1
) -> List[Dict[str, Any]]:
    """
    Calculates valid event counts and valid time for respective windows.

    The primary algorithm uses a two-pointer sweep approach across the sorted data
    to ensure O(N_events + N_gap) efficiency.

    Args:
        gaps: Consolidated, sorted list of (start, end] tuples (timestamps in seconds).
        event_timestamps: Sorted list of event timestamps (in seconds).
        num_series: number of series for number of events

    Returns:
        A list of dictionaries, one for each 5-minute window, with the calculated metrics.
    """
    if not event_timestamps and not gaps:
        return []

    # 1. Determine the overall time range and window alignment
    
    # Find the absolute earliest and latest time in the data
    min_time = 0
    max_time = max(gaps[-1].upper if gaps else 0, event_timestamps[-1])

    # Align the starting time to the nearest earlier 10-minute mark
    start_time_base = 0
    
    # Calculate the number of 5-minute windows needed to cover the range
    # Ensure all data is covered, rounding up to the nearest window
    num_windows = (max_time - start_time_base + WINDOW_SIZE - 1) // WINDOW_SIZE
    
    # Initialize pointers for efficient sweep
    gap_index = 0
    event_index = 0
    results = []

    # 2. Sweep through each 10-minute window
    for k in range(int(num_windows)):
      window_start = start_time_base + k * WINDOW_SIZE
      window_end = window_start + WINDOW_SIZE
      
      # Lists to capture data relevant to the current window
      window_gap_overlaps = []
      invalid_event_count = 0
      total_event_count = 0
      
      # --- PART 1: Calculate Valid Time (Seconds) ---
      
      # Advance gap_index to the first interval that can potentially overlap the window
      # We only advance the main gap_index once per window sweep.
      while gap_index < len(gaps) and gaps[gap_index].upper <= window_start:
        gap_index += 1

      # Now, gap.upper is later than window_start
      # Check for overlaps within the current window and clip/collect them
      current_gap_index = gap_index
      while current_gap_index < len(gaps):
        gap_start = gaps[current_gap_index].lower
        gap_end = gaps[current_gap_index].upper
        
        # Optimization: If the current gap interval starts past the window end, we stop.
        if gap_start >= window_end:
            break

        # Check if there is an overlap (s < we AND e > ws)
        if gap_end > window_start:
          # Clip the gap interval to the window boundaries
          clip_start = max(gap_start, window_start)
          clip_end = min(gap_end, window_end)
          
          if clip_end > clip_start:
            window_gap_overlaps.append(P.closedopen(clip_start, clip_end))

        current_gap_index += 1

      # Calculate the total gap duration
      total_gaps = sum([gap.upper - gap.lower for gap in window_gap_overlaps]) /sampling_rate
      valid_seconds = min(max_time - k * WINDOW_SIZE, WINDOW_SIZE)/sampling_rate - total_gaps
      if k == 0:
        # adjust the warmup period for the first window
        valid_seconds -= 30
      
      # --- PART 2: Count Valid Events ---

      # Advance event_index to the first event that is inside or after the window start
      while event_index < len(event_timestamps) and event_timestamps[event_index] < window_start:
        event_index += 1

      # Process all events within the current window [window_start, window_end)
      current_event_index = event_index
      
      # The gap pointer (gap_index) is already positioned at the start of relevant intervals
      current_gap_index = gap_index
      
      while current_event_index < len(event_timestamps):
        event_ts = event_timestamps[current_event_index]
        
        # Stop if the event is past the window end
        if event_ts >= window_end:
          break
            
        total_event_count += 1
        is_valid_event = True
        
        # Advance the gap pointer until its end is > event_ts
        while current_gap_index < len(gaps) and gaps[current_gap_index].upper < event_ts:
          current_gap_index += 1
        
        # Check the current gap interval I_j = (s_j, e_j]
        if current_gap_index < len(gaps) and event_ts in gaps[current_gap_index]:
          invalid_event_count += 1
        
        current_event_index += 1
          
      # 3. Store results for the current window
      # Only append results if there was activity (events or gap time) in the window
      if valid_seconds > 0.0:
        bpm = (total_event_count - invalid_event_count) / valid_seconds * 60 / num_series
      else:
        bpm = 0

      if total_event_count > 0 or total_gaps > 0:
        results.append({
            "window_start_ts": window_start,
            "window_start_ts_in_min": window_start/sampling_rate/60,
            "valid_seconds": valid_seconds,
            "gap_seconds": total_gaps,
            "invalid_event_count": invalid_event_count,
            "total_event_count": total_event_count,
            "num_series": num_series,
            "bpm": bpm,
        })
            
    return results

# --- Sample Data and Execution ---

# Unix timestamps (seconds since epoch) for clarity.
# Example starting point: 2024-01-01 10:00:00 UTC (1704096000)

# The total range is set up to cover 10:00 to 10:40 (4 windows).
BASE_TS = 1704096000 # 2024-01-01 10:00:00

# Two separate lists of gap intervals, both sorted. 
INVALID_LIST_A = [
    (BASE_TS + 300, BASE_TS + 480),   # 10:05:00 to 10:08:00
    (BASE_TS + 1500, BASE_TS + 2100), # 10:25:00 to 10:35:00 
    (BASE_TS + 2040, BASE_TS + 2160), # 10:34:00 to 10:36:00
]

INVALID_LIST_B = [
    (BASE_TS + 900, BASE_TS + 1200),  # 10:15:00 to 10:20:00
    (BASE_TS + 1700, BASE_TS + 2050), # 10:28:20 to 10:34:10 
    (BASE_TS + 2200, BASE_TS + 2300)  # 10:36:40 to 10:38:20 
]

# Event timestamps (instants in time)
SAMPLE_EVENTS = [
    # Window 1: [10:00:00, 10:10:00)
    BASE_TS + 100, # V (10:01:40)
    BASE_TS + 350, # I (10:05:50)
    BASE_TS + 480, # I (10:08:00)
    BASE_TS + 500, # V (10:08:20)
    
    # Window 2: [10:10:00, 10:20:00)
    BASE_TS + 650, # V (10:10:50)
    BASE_TS + 901, # I (10:15:01)
    BASE_TS + 1000, # I (10:16:40)
    BASE_TS + 1200, # V (10:20:00)
    
    # Window 3: [10:20:00, 10:30:00)
    BASE_TS + 1300, # V (10:21:40)
    BASE_TS + 1501, # I (10:25:01)
    BASE_TS + 1800, # I (10:30:00)
    
    # Window 4: [10:30:00, 10:40:00)
    BASE_TS + 1900, # I (10:31:40)
    BASE_TS + 2150, # I (10:35:50)
    BASE_TS + 2200, # I (10:36:40)
    BASE_TS + 2250, # I (10:37:30)
    BASE_TS + 2350, # V (10:39:10)
]

def plot_bpm(result, fname):
  fig, ax = plt.subplots(figsize=(12, 7))
  ax.set_title('Comparison of BPM and Gap Seconds by Method: ' + fname)
  ax.set_xlabel('Minutes')
  ax.set_ylabel('BPM (Line) / Zero Line (Gap Length)')

  method_index = 0
  y_max = 0
  color = plt.get_cmap("rainbow")(np.linspace(0, 1, num=len(result)))
  step_size = 4
  for method in result:
    # Setup common X-axis components
    df = pd.DataFrame(result[method])
    # minutes
    xtics_numeric = df["window_start_ts"] / sampling_rate / 60
    xtics_str = map(str, xtics_numeric[::step_size])
    y_zeros = np.zeros_like(xtics_numeric, dtype=float)
    y_max = min(max(y_max, df["bpm"].max()), 200)


    # BPM Line (Solid line, circle marker)
    ax.plot(xtics_numeric, df['bpm'], marker='o', linestyle='-',
            color=color[method_index], label=f'{method}')

    # Gap Seconds Segment (Error Bar) 
    x_err_pan = (df['gap_seconds']) / 2 / 60
    ax.errorbar(
        x=xtics_numeric, y=y_zeros + method_index - 4, xerr=x_err_pan,
        fmt='none', ecolor=color[method_index], capsize=0, elinewidth=2)
    method_index += 1

  # --- Final Plot Adjustments ---
  # Set x-axis ticks
  ax.set_xticks(xtics_numeric[::step_size])
  ax.set_xticklabels(xtics_str)

  # Adjust y-limits
  ax.set_ylim(-5.5, y_max * 1.1)

  ax.grid(True, axis='y', linestyle='--', alpha=0.6)
  # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
  ax.legend(loc='upper right')
  plt.tight_layout()
  plt.show(block=False)
  plt.savefig("./../plots/" + fname + ".png")

def flatten_data(data, num_cols):
  # data may have an integer multiple of num_col  
  # this function keeps the first num_col and concatenates the rest to the first
  # num_col entries.
  for index in range(num_cols, len(data)):
    # the list count should be N times num_cols 
    col_index = index % num_cols
    data[col_index] += data[index]

def get_representative_item(window_start, items):
  item = { "window_start_ts": window_start,
           "window_start_ts_in_min": window_start/sampling_rate/60,
           "gap_seconds": 0,
           "num_series": 0,
           "bpm": 0,
        }

  bpms = np.array([item["bpm"] for item in items])
  similar = compare_three_values(bpms)
  print("num(similar): " + str(sum(similar)))
  snrs= np.array([item["snr"] for item in items])
  print("bpms: " + str(bpms))
  print("snrs: " + str(snrs))
  if sum(similar) >= 2:
    item["num_series"] = sum(similar)
    item["bpm"] = bpms[similar].sum() / sum(similar)
    item["gap_seconds"] = np.where(similar, [item["gap_seconds"] for item in
                                             items], 0).sum() / sum(similar)
  else:
    max_snr_index = np.argmax(snrs)
    item["num_series"] = 1
    item["bpm"] = 0
    item["gap_seconds"] = items[max_snr_index]["gap_seconds"]

  pprint.pprint(item)
  return item



if __name__ == "__main__":
  # fname="all_data_20240322_P21_1-00_00_00_000_05_59_59_998.ascii_30.0x1scipy.npz"

  parser = argparse.ArgumentParser(description="Peak detection script using scipy.signal.find_peaks")
  parser.add_argument("--inputfile", type=str, default="all_data.npz", help="Path to save the results")
  parser.add_argument("--input_dir", type=str, default=".", help="Directory for processing")

  args = parser.parse_args()

  fname=os.path.join(args.input_dir, args.inputfile)
  print(fname)
  all_data = np.load(fname, allow_pickle=True)
  # gaps = all_data["gaps"].item()
  rpeaks = all_data["rpeaks"].item()
  snr = all_data["snr"].item()
  total_sec = all_data["total_sec"].item()
  snr_window = all_data["snr_window"].item()
  snr_increment = snr_window * sampling_rate // WINDOW_SIZE
  print("snr_increment: " + str(snr_increment))
  print("total-sec: " + str(total_sec))

  result = {}

  for method in rpeaks.keys():
    num_cols = len(rpeaks[method])
    print("method: " + method + " num_cols: " + str(num_cols))
    result[method] = []

    # data may have an integer multiple of num_col, so merge them using module
    # operation
    flatten_data(rpeaks[method], num_cols)
    rpeaks[method] = rpeaks[method][:num_cols]
    bpm = [0] * num_cols

    gaps = [[] for _ in range(num_cols)]
    for col_index in range(num_cols):
      gaps[col_index] = find_gaps(rpeaks[method][col_index])
      total_gap = sum([gap.upper-gap.lower for gap in gaps[col_index]]) / sampling_rate
      num_rpeaks = len(rpeaks[method][col_index]) - len(gaps[col_index])
      bpm[col_index] = num_rpeaks / (total_sec - total_gap) * 60

      print("col_index: " + str(col_index) + " total_gap in sec: " + str(total_gap) +
            " #rpeaks: " + str(num_rpeaks))

    #########################
    # 1. First, pre-calculate metrics for all leads and group them by window
    windows_registry = {} # { timestamp: [list_of_lead_metrics] }
    for i in range(num_cols):
      # append one more snr value, to handle boundary condition
      snr[method][i].append(snr[method][i][-1])
      snr_index = 0
      for item in calculate_time_series_metrics(gaps[i], rpeaks[method][i]):
        ts = item["window_start_ts"] 
        item["snr"] = snr[method][i][snr_index // snr_increment]
        snr_index += 1
        if ts not in windows_registry:
          windows_registry[ts] = []
        windows_registry[ts].append(item)

    # 2. Now process one time interval at a time
    for ts in sorted(windows_registry.keys()):
      item = get_representative_item(ts, windows_registry[ts])
      if item:
        result[method].append(item)

    ##########################

    #pprint.pprint(gaps)
    # pprint.pprint(bpm)
    # pprint.pprint(snr)

  if True:
    plot_bpm(result, all_data["inputfile"])

def old_code():
      # missing while loop
      similar = compare_three_values(bpm)
      print("\nsimilar for method: " + method + " ->  " + str(similar))
      if sum(similar) == 0 or sum(similar) == 1:
        print("Three measurements did not agree when using method: " + method)

      use_weighted_avg = False
      num_series = sum(similar)

      # pre-processing
      temp_result = {} 
      result[method] = []
      keys_to_merge = ["valid_seconds", "gap_seconds", "invalid_event_count",
                       "total_event_count", "total_bpm_sum"]
      for i in range(len(similar)):
        if not similar[i]:
          continue
        # combine seconds and rpeaks
        for item in calculate_time_series_metrics(gaps[i], rpeaks[method][i]):
          ts = item["window_start_ts"]
          if item["valid_seconds"] > 0.0:
            item["total_bpm_sum"] = (item["total_event_count"] - item["invalid_event_count"]) / item["valid_seconds"] * 60 
          else:
            item["total_bpm_sum"] = 0

          if ts in temp_result:
            for k in keys_to_merge:
              temp_result[ts][k] += item[k]
          else:
            temp_result[ts] = item

      if use_weighted_avg:
        for ts in sorted(temp_result.keys()):
          # recalculate bpm
          item = temp_result[ts]
          if item["valid_seconds"] > 0.0:
            bpm = (item["total_event_count"] - item["invalid_event_count"]) / item["valid_seconds"] * 60 
          else:
            bpm = 0
          item["bpm"] = bpm
          item["gap_seconds"] /=  num_series
          item["num_series"] = num_series
          result[method].append(item)
      else:
        for ts in sorted(temp_result.keys()):
          # recalculate bpm
          item = temp_result[ts]
          bpm = item["total_bpm_sum"] / num_series
          item["bpm"] = bpm
          item["gap_seconds"] /=  num_series
          item["num_series"] = num_series
          result[method].append(item)

