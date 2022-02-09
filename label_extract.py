import pandas as pd
import numpy as np
from scipy.io import savemat
from os import walk

filenames = next(walk('./data/downsample/'), (None, None, []))[2]  # [] if no file
f1 = pd.read_csv('C:/wzw/data/NICU/Annotations/annotations_2017_A.csv', index_col=False)
f2 = pd.read_csv('C:/wzw/data/NICU/Annotations/annotations_2017_B.csv', index_col=False)
f3 = pd.read_csv('C:/wzw/data/NICU/Annotations/annotations_2017_C.csv', index_col=False)

arr_1 = f1.to_numpy()
arr_2 = f2.to_numpy()
arr_3 = f3.to_numpy()
out = np.zeros((15416, 79))

# last_seizure = -1

for j in range(79):
	ratio21 = 0
	ratio12 = 0
	ratio30 = 0
	ratio03 = 0
	for i in range(15416):
		cnt1 = 0  # count1
		cnt2 = 0  # count2
		if arr_1[i][j] == 1:
			'''
			if (i - last_seizure) > 600:
				for preictal in range(1,601):
					arr_1[i-preictal][j] = 2
			last_seizure = i
			
			'''
			cnt1 += 1
		elif arr_1[i][j] == 0:
			cnt2 += 1
		if arr_2[i][j] == 1:
			cnt1 += 1
		elif arr_2[i][j] == 0:
			cnt2 += 1
		if arr_3[i][j] == 1:
			cnt1 += 1
		elif arr_3[i][j] == 0:
			cnt2 += 1
		# nancounter = 0
		# if np.isnan(arr_2[i][j]):
		# 	nancounter += 1
		# if np.isnan(arr_3[i][j]):
		# 	nancounter += 1
		# if np.isnan(arr_1[i][j]):
		# 	nancounter += 1
		# if nancounter != 0 and nancounter != 3:
		# 	print(i, j)
		# TODO change to 2
		# if at least two experts agree that label is 1
		if cnt1 >= 2:
			out[i, j] = 1
		# if cnt2 >= 2:
		# 	out[i, j] = 0
		if cnt1 == 2:
			ratio21 += 1
		if cnt2 == 2:
			ratio12 += 1
		if cnt1 == 3:
			ratio30 += 1
		if cnt2 == 3:
			ratio03 += 1
	print(j+1, ratio03, ratio12, ratio21, ratio30,)
	if ratio30 >= 1000:
		print(j+1) # 30
	# input('input something')
savemat('data/downsample/eeg_label79' + '.mat', {'y': out})
savemat('data/downsample/eeg_label10' + '.mat', {'y': out[:, [4, 13, 18, 37, 38, 40, 65, 66, 68, 77]]})


# # TODO split by 10 seconds
