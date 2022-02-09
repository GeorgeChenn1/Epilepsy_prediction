import json
import mne
import numpy as np
import scipy.io
from scipy.io import savemat, loadmat
from os import walk

filenames = next(walk('C:/wzw/data/NICU/EEG/'), (None, None, []))[2]  # [] if no file
print(filenames)

root_dir = 'C:/wzw/data/NICU/EEG/'



for i in range(len(filenames)):
    curr_file_path = root_dir + filenames[i]
    raw_data = mne.io.read_raw_edf(curr_file_path, preload=True)
    eeg = raw_data.to_data_frame()
    np_arr = eeg.to_numpy()
    # print(np_arr.shape)
    # print(type(np_arr.shape))
    # input('')
    concatted = []
    for j in range((np_arr.shape[0] // 256) - 1):
        temp = np_arr[j*4: (j+1)*4, :]  # mean(256) → mean(4)
        assert temp.shape == (4, 22)  # assert means if not(event) alert
        averaged = np.mean(temp, axis=0)
        concatted.append(averaged.reshape(1, 22))
    out = np.concatenate(concatted, axis=0)
    path_label = './data/downsample/eeg_label79.mat'
    data = scipy.io.loadmat(path_label)['y']
    t = int(filenames[i].replace('eeg', '').replace('.edf', ''))
    print('!!!', t)

    labels = data[:, t-1]
    labels = labels.reshape(-1, 1)
    labels = labels[:out.shape[0]]

    savemat('data/downsample/' + filenames[i].replace('.edf', '') + '_ds.mat', {'x': out, 'y': labels})
    print(filenames[i])

