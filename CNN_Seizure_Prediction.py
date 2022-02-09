# -*- coding:utf-8 -*-
"""
作者：WZW
日期：2021年05月18日
CNN跑Kaggle癫痫预测数据集，预计效果Test loss: 0.12622499525547026
Test accuracy: 0.95166665
跑30个epoch*21000
"""
from __future__ import print_function
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import random
import numpy as np
import scipy.io
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#Visualizing an example:
interictal_tst = 'D:/SeizureData/Patient_1_interictal_segment_0001.mat'
preictal_tst = 'D:/SeizureData/Patient_1_preictal_segment_0001.mat'
interictal_data = scipy.io.loadmat(interictal_tst)
preictal_data = scipy.io.loadmat(preictal_tst)

interictal_array = interictal_data['interictal_segment_1'][0][0][0]
preictal_array = preictal_data['preictal_segment_1'][0][0][0]

# Taking 1 second worth of data of channel 0 for preictal example
l = list(range(10000))
for i in l[::5000]:
    print('Interictal')
    i_secs = interictal_array[0][i:i+5000]
    i_f, i_t, i_Sxx = spectrogram(i_secs, fs=5000, return_onesided=False)
    i_SS = np.log1p(i_Sxx)
    plt.imshow(i_SS[:] / np.max(i_SS), cmap='gray')
    plt.title('发作间期(Interictal)')
    plt.show()
    print('Preictal')
    p_secs = preictal_array[0][i:i+5000]
    p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
    p_SS = np.log1p(p_Sxx)
    plt.imshow(p_SS[:] / np.max(p_SS), cmap='gray')
    plt.title('发作前期(preictal)')
    plt.show()


# Creating training and testing data
all_X = []
all_Y = []

types = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']

for i,typ in enumerate(types):
    # Looking at 18 files for each event for a balanced dataset
    for j in range(18):
        fl = 'Patient_1/{}_{}.mat'.format(typ, str(j + 1).zfill(4))
        data = scipy.io.loadmat(fl)
        k = typ.replace('Patient_1_', '') + '_'
        d_array = data[k + str(j + 1)][0][0][0]
        lst = list(range(3000000))  # 10 minutes
        for m in lst[::5000]:
            # Create a spectrogram every 1 second
            p_secs = d_array[0][m:m+5000]
            p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
            p_SS = np.log1p(p_Sxx)
            arr = p_SS[:] / np.max(p_SS)
            all_X.append(arr)
            all_Y.append(i)


# Shuffling the data
dataset = list(zip(all_X, all_Y))
random.shuffle(dataset)
all_X,all_Y = zip(*dataset)
print(len(all_X))
# Splitting data into train/test, leaving only 600 samples for testing
x_train = np.array(all_X[:21000])
y_train = np.array(all_Y[:21000])
x_test = np.array(all_X[21000:])
y_test = np.array(all_Y[21000:])
batch_size = 128
num_classes = 2
epochs = 30
img_rows, img_cols = 256, 22
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Formatting the labels for training
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
