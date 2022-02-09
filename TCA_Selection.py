# -*- coding:utf-8 -*-
"""
作者：dell
日期：2021年05月23日
"""
import scipy.io
import scipy
import scipy.linalg
import sklearn.metrics
import numpy as np
import pandas as pd
import scipy.io as sio
import pywt
import scipy.stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt  # 绘图库
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['FangSong']  # 可显示中文字符
plt.rcParams['axes.unicode_minus'] = False


# 绘制confusion matrix图像函数
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


# 核函数
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA_1:
    # TCA算法
    # 输入 核类型（三种可选）、降低到的维度、超参数lambda和gamma
    def __init__(self, kernel_type='rbf', dim=30, lamb=0.1, gamma=0.1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new


# 计算熵特征（非线性）

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


# 计算交叉特征

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


# 三种特征合并

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


# 划分训练集和测试集

def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 7:3)
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


# Dog_3训练集
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

# Dog_2测试集
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
X_train_TL, X_test_TL = TCA_1().fit(X_train, X_test1)

# GBT和RF分类
cls = GradientBoostingClassifier(n_estimators=100)
cls.fit(X_train_TL, Y_train)
train_score = cls.score(X_train_TL, Y_train)
test_score = cls.score(X_test_TL, Y_test1)
Y_pre1 = cls.predict(X_test_TL)
print("跨患者+TCA GBT（梯度提升树）训练集上准确度为： {}".format(train_score))
print("跨患者+TCA GBT（梯度提升树）测试集上准确度为： {}".format(test_score))
cm1 = np.array(confusion_matrix(Y_test1, Y_pre1))  # 将list类型转换成数组类型，如果已经是numpy数组类型，则忽略此步骤。
labels_name = ['inter-ictal', 'pre-ictal']  # 这里给横纵坐标标签集合赋值
plot_confusion_matrix(cm1, labels_name, "TCA+GBT confusion matrix")  # 调用函数
print('GBT prediction score：', accuracy_score(Y_test1, Y_pre1))
print('confusion matrix:', confusion_matrix(Y_test1, Y_pre1))

cls = RandomForestClassifier(n_estimators=100)
cls.fit(X_train_TL, Y_train)
train_score = cls.score(X_train_TL, Y_train)
test_score = cls.score(X_test_TL, Y_test1)
Y_pre2 = cls.predict(X_test_TL)
print("跨患者+TCA RF（随机森林）训练集上准确度为： {}".format(train_score))
print("跨患者+TCA RF（随机森林）测试集上准确度为： {}".format(test_score))
cm2 = np.array(confusion_matrix(Y_test1, Y_pre2))  # 将list类型转换成数组类型，如果已经是numpy数组类型，则忽略此步骤。
labels_name = ['inter-ictal', 'pre-ictal']  # 这里给横纵坐标标签集合赋值
plot_confusion_matrix(cm2, labels_name, "TCA+RF confusion_matrix")  # 调用函数
print('RF prediction score：', accuracy_score(Y_test1, Y_pre2))
print('confusion matrix:', confusion_matrix(Y_test1, Y_pre2))
