# -*- coding:utf-8 -*-
"""
作者：dell
日期：2021年05月27日
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt  # 绘图库
import pywt
import scipy.stats
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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


# 计算熵特征
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


# 特征合并
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics


# 划分训练集测试集
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


# 训练集+测试集8:2
filename = 'D:/SeizureData/3/EEG_Dog_3_480.mat'
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

df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, ycol, xcols, ratio=0.8)
cls = GradientBoostingClassifier(n_estimators=1000)
cls.fit(X_train, Y_train)
train_score = cls.score(X_train, Y_train)
test_score = cls.score(X_test, Y_test)
Y_pre1 = cls.predict(X_test)
print("GBT（梯度提升树）训练集上准确度为： {}".format(train_score))
print("GBT（梯度提升树）测试集上准确度为： {}".format(test_score))
cm1 = np.array(confusion_matrix(Y_test, Y_pre1))  # 将list类型转换成数组类型，如果已经是numpy数组类型，则忽略此步骤。
labels_name = ['发作间期', '发作前期']  # 这里给横纵坐标标签集合赋值
plot_confusion_matrix(cm1, labels_name, "GBT混淆矩阵(confusion_matrix)")  # 调用函数
print('GBT分类器预测得分为：', accuracy_score(Y_test, Y_pre1))
print('混淆矩阵为:', confusion_matrix(Y_test, Y_pre1))

cls = RandomForestClassifier(n_estimators=1000)
cls.fit(X_train, Y_train)
train_score = cls.score(X_train, Y_train)
test_score = cls.score(X_test, Y_test)
Y_pre2 = cls.predict(X_test)
print("RF（随机森林）训练集上准确度为： {}".format(train_score))
print("RF（随机森林）测试集上准确度为： {}".format(test_score))
# confusion_matrix(Y_test, Y_pre)# 绘制混淆矩阵
# accuracy_score(Y_test, Y_pre)# 计算得分
cm2 = np.array(confusion_matrix(Y_test, Y_pre2))  # 将list类型转换成数组类型，如果已经是numpy数组类型，则忽略此步骤。
labels_name = ['发作间期', '发作前期']  # 这里给横纵坐标标签集合赋值
plot_confusion_matrix(cm2, labels_name, "RF混淆矩阵(confusion_matrix)")  # 调用函数
print('RF分类器预测得分为：', accuracy_score(Y_test, Y_pre2))
print('混淆矩阵为:', confusion_matrix(Y_test, Y_pre2))
