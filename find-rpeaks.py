
# Load NeuroKit and other useful packages
# /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/neurokit2
import neurokit2 as nk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# uses qt5agg for interactive display.
matplotlib.use('qt5agg')
cols_index = (1, 2, 3)
cols_name = ('B', 'C', 'D')

cols = np.loadtxt("../../pig-data/pig24-D82-file-1-shorter.csv", delimiter=",",
                  usecols=cols_index, skiprows=11)

combined_signals = pd.DataFrame()
combined_rpeaks = []
for i in range(0, len(cols_index)):
  # Automatically process the (raw) ECG signal
  signals, info = nk.ecg_process(cols[:,i], sampling_rate=500)
  combined_signals.insert(i, "col" + cols_name[i], signals["ECG_Clean"])
  combined_rpeaks += [info["ECG_R_Peaks"]]
  nk.ecg_plot(signals, info);
  plt.savefig("plots/ecg-details-col" + cols_name[i] + ".png")
  # plt.show(block=False)

# 2025-10-26 averaging didn't really work
# next step: threshold based

# Visualize R-peaks in ECG signal
#plot = nk.events_plot(rpeaks, cleaned_ecg)
plot = nk.events_plot(combined_rpeaks, combined_signals)
plot.show()
# fig = plt.gcf() fig.set_size_inches(10, 12, forward=True) fig.savefig(“myfig.png”)
#nk.ecg_plot(signals, info);
#plt.show(block=False)
