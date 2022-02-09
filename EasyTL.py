# -*- coding:utf-8 -*-
"""
作者：dell
日期：2021年05月23日
"""
from intra_alignment import CORAL_map
from label_prop_v2 import label_prop
import scipy.io
import scipy
import scipy.linalg
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import pywt
import scipy.stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt  # 绘图库
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['FangSong']  # 可显示中文字符
plt.rcParams['axes.unicode_minus'] = False


# 绘制混淆矩阵
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # 获取标签的间隔数
    plt.xticks(num_class, labels_name, rotation=0)  # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)  # 将标签印在y轴坐标上

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 求cos距离特征
def get_cosine_dist(A, B):
    B = np.reshape(B, (1, -1))

    if A.shape[1] == 1:
        A = np.hstack((A, np.zeros((A.shape[0], 1))))
        B = np.hstack((B, np.zeros((B.shape[0], 1))))

    aa = np.sum(np.multiply(A, A), axis=1).reshape(-1, 1)
    bb = np.sum(np.multiply(B, B), axis=1).reshape(-1, 1)
    ab = A @ B.T

    # to avoid NaN for zero norm
    aa[aa == 0] = 1
    bb[bb == 0] = 1

    D = np.real(np.ones((A.shape[0], B.shape[0])) - np.multiply((1 / np.sqrt(np.kron(aa, bb.T))), ab))

    return D


# 求ma_distance特征
def get_ma_dist(A, B):
    Y = A.copy()
    X = B.copy()

    S = np.cov(X.T)
    try:
        SI = np.linalg.inv(S)
    except:
        print("Singular Matrix: using np.linalg.pinv")
        SI = np.linalg.pinv(S)
    mu = np.mean(X, axis=0)

    diff = Y - mu
    Dct_c = np.diag(diff @ SI @ diff.T)

    return Dct_c


# 求取类别中心
def get_class_center(Xs, Ys, Xt, dist):
    source_class_center = np.array([])
    Dct = np.array([])
    for i in np.unique(Ys):
        sel_mask = Ys == i
        X_i = Xs[sel_mask.flatten()]
        mean_i = np.mean(X_i, axis=0)
        if len(source_class_center) == 0:
            source_class_center = mean_i.reshape(-1, 1)
        else:
            source_class_center = np.hstack((source_class_center, mean_i.reshape(-1, 1)))

        if dist == "ma":
            Dct_c = get_ma_dist(Xt, X_i)
        elif dist == "euclidean":
            Dct_c = np.sqrt(np.nansum((mean_i - Xt) ** 2, axis=1))
        elif dist == "sqeuc":
            Dct_c = np.nansum((mean_i - Xt) ** 2, axis=1)
        elif dist == "cosine":
            Dct_c = get_cosine_dist(Xt, mean_i)
        elif dist == "rbf":
            Dct_c = np.nansum((mean_i - Xt) ** 2, axis=1)
            Dct_c = np.exp(- Dct_c / 1);

        if len(Dct) == 0:
            Dct = Dct_c.reshape(-1, 1)
        else:
            Dct = np.hstack((Dct, Dct_c.reshape(-1, 1)))

    return source_class_center, Dct


# 定义EasyTL特征对齐方式
def EasyTL(Xs, Ys, Xt, Yt, intra_align="coral", dist="euclidean", lp="linear"):
    # Inputs:
    #   Xs          : source data, ns * m
    #   Ys          : source label, ns * 1
    #   Xt          : target data, nt * m
    #   Yt          : target label, nt * 1
    # The following inputs are not necessary
    #   intra_align : intra-domain alignment: coral(default)|gfk|pca|raw
    #   dist        : distance: Euclidean(default)|ma(Mahalanobis)|cosine|rbf
    #   lp          : linear(default)|binary

    # Outputs:
    #   acc         : final accuracy
    #   y_pred      : predictions for target domain

    # Reference:
    # Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang.
    # Easy Transfer Learning By Exploiting Intra-domain Structures.
    # IEEE International Conference on Multimedia & Expo (ICME) 2019.

    C = len(np.unique(Ys))
    if C > np.max(Ys):
        Ys += 1
        Yt += 1

    m = len(Yt)

    if intra_align == "raw":
        print('EasyTL using raw feature...')
    elif intra_align == "pca":
        print('EasyTL using PCA...')
        print('Not implemented yet, using raw feature')
    # Xs, Xt = PCA_map(Xs, Xt)
    elif intra_align == "gfk":
        print('EasyTL using GFK...')
        print('Not implemented yet, using raw feature')
        # Xs, Xt = GFK_map(Xs, Xt)
    elif intra_align == "coral":
        print('EasyTL using CORAL...')
        Xs = CORAL_map(Xs, Xt)

    _, Dct = get_class_center(Xs, Ys, Xt, dist)
    print('Start intra-domain programming...')
    Mcj = label_prop(C, m, Dct, lp)
    y_pred = np.argmax(Mcj, axis=1) + 1
    acc = np.mean(y_pred == Yt.flatten());

    return acc, y_pred


# 计算非线性特征（熵）
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


# 计算统计特征
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


# 计算几何特征（交叉线）
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


# 将特征组合
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


# 划分训练集测试集
def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test


# 训练集dog2
filename = 'data/3/EEG_Dog_3_480.mat'
eeg_data = sio.loadmat(filename)
eeg_signals = eeg_data['EEGData'][0][0][0]
eeg_labels_ = eeg_data['EEGData'][0][0][1]
eeg_labels = list(map(lambda x: x[0][0], eeg_labels_))

dict_eeg_data = defaultdict(list)
for ii, label in enumerate(eeg_labels):
    dict_eeg_data[label].append(eeg_signals[ii])
list_labels = []
list_features = []
for k, v in dict_eeg_data.items():
    yval = list(dict_eeg_data.keys()).index(k)
    for signal in v:
        features = []
        list_labels.append(yval)
        list_coeff = pywt.wavedec(signal, 'sym5')
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
df = pd.DataFrame(list_features)

ycol = 'y'
xcols = list(range(df.shape[1]))
df.loc[:, ycol] = list_labels
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, ycol, xcols, ratio=1)

# 新加代码段

# 测试集dog3
filename = 'data/2/EEG_480.mat'
eeg_data = sio.loadmat(filename)
eeg_signals = eeg_data['EEGData'][0][0][0]
eeg_labels_ = eeg_data['EEGData'][0][0][1]
eeg_labels = list(map(lambda x: x[0][0], eeg_labels_))

dict_eeg_data = defaultdict(list)
for ii, label in enumerate(eeg_labels):
    dict_eeg_data[label].append(eeg_signals[ii])
list_labels = []
list_features = []
for k, v in dict_eeg_data.items():
    yval = list(dict_eeg_data.keys()).index(k)
    for signal in v:
        features = []
        list_labels.append(yval)
        list_coeff = pywt.wavedec(signal, 'sym5')
        for coeff in list_coeff:
            features += get_features(coeff)
        list_features.append(features)
df1 = pd.DataFrame(list_features)
ycol1 = 'y'
xcols1 = list(range(df1.shape[1]))
df1.loc[:, ycol1] = list_labels

df_train, df_test, X_train1, Y_train1, X_test1, Y_test1 = get_train_test(df1, ycol1, xcols1, ratio=0)
# 新的训练集和测试集
list_acc = []
t0 = time.time()
Acc1, y_pred1 = EasyTL(X_train, Y_train, X_test1, Y_test1, "raw")
t1 = time.time()
print("Time Elapsed: {:.2f} sec".format(t1 - t0))
Acc2, y_pred2 = EasyTL(X_train, Y_train, X_test1, Y_test1)
t2 = time.time()
print("Time Elapsed: {:.2f} sec".format(t2 - t1))

print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1 * 100, Acc2 * 100))
list_acc.append([Acc1, Acc2])
acc = np.array(list_acc)
avg = np.mean(acc, axis=0)
print('EasyTL(c) AVG Acc: {:.1f} %'.format(avg[0] * 100))
print('EasyTL AVG Acc: {:.1f} %'.format(avg[1] * 100))
print(y_pred1)
print(y_pred2)
file_name = 'data.mat'
sio.savemat(file_name, {"data": y_pred2})

cm = np.array(confusion_matrix(Y_test1, y_pred2))  # 将list类型转换成数组类型，如果已经是numpy数组类型，则忽略此步骤。
labels_name = ['inter-ictal', 'pre-ictal']  # 这里给横纵坐标标签集合赋值
plot_confusion_matrix(cm, labels_name, "EasyTL confusion matrix)")  # 调用函数
print('RF分类器预测得分为：', accuracy_score(Y_test1, y_pred2))
print('混淆矩阵为:', confusion_matrix(Y_test1, y_pred2))
