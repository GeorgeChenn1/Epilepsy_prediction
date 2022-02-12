import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from os import walk

filenames = next(walk('D:/wzw/DSAN/seizure5times/data/'), (None, None, []))[2]  # [] if no file
print(filenames)
root_dir = 'D:/wzw/DSAN/seizure5times/data/'

f = []
for i in range(len(filenames)):
    curr_file_path = root_dir + filenames[i]
    m = loadmat(curr_file_path)
    arr = m['x']
    f.append(arr)

for i in range(len(filenames)):
    src = f.copy()
    tar = src.pop(i)
    # TODO split valid set
    index_train = int(0.6 * tar.shape[0])
    # index_test = tar.shape[0] -  int(0.6*tar.shape[0])
    tar_train = tar[0:index_train, :, :]
    tar_test = tar[index_train: tar.shape[0], :, :]
    print(tar.shape, tar_train.shape, tar_test.shape)
    input('')
#    print(tar.shape)



# result = np.concatenate((arr1, arr2, arr3), axis=0)
# print(result.shape)
# input('')
