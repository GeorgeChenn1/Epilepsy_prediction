
import pyedflib
import numpy as np

f = pyedflib.EdfReader('C:/wzw/data/NICU/EEG/eeg2.edf')
n = f.signals_in_file
print("signal numbers:", n)
signal_labels = f.getSignalLabels()
print("Labels:", signal_labels)
signal_headers = f.getSignalHeaders()
print("Headers:", signal_headers)