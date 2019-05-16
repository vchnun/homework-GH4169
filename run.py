
# -*- encoding: utf-8 -*-
# @File        :   run.py
# @Contact     :   vchnun@gmail.com
# @Time        :   2019/5/10 10:34
# @Author      :   chenwang
# @Version     :   0.1
# @Description :   None

import os
import numpy as np
from sklearn.model_selection import cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
# 方法选择
# 决策树回归 ExtraTree极端随机数回归
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
# 线性回归
from sklearn.linear_model import LinearRegression
# SVM回归
from sklearn.svm import SVR
# kNN回归
from sklearn.neighbors import KNeighborsRegressor
# 随机森林回归 Adaboost回归 GBRT回归 Bagging回归
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor


def run():
    data, gt = read_data()
    methods = (LinearRegression, DecisionTreeRegressor, KNeighborsRegressor, SVR, ExtraTreeRegressor,
               RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor)
    # args = [50, 'auto']  # 一些模型的参数设置
    all_score = []
    for method in methods:
        reg = method()
        # 十折交叉验证
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=10)
        scores = cross_val_score(reg, data, gt.ravel(), cv=cv, scoring='neg_mean_squared_error')
        np.set_printoptions(precision=5, formatter={'float': '{:.3e}'.format})
        print(method.__name__, np.array(0-scores))
        all_score.append(np.mean(scores))

    for i, method in enumerate(methods):
        print("方法{}的分数为：{:.4e}".format(method.__name__, 0-all_score[i]))

    eval_the(LinearRegression, data, gt)
    '''
    # 画出全部图像
    plt.figure(1)
    plt.rcParams['figure.figsize'] = (9.0, 12.0)
    plt.rcParams['figure.dpi'] = 300

    for i, method in enumerate(methods[1:]):
        result, x_gt, y = eval_the(method, data, gt)
        plt.subplot(4, 2, i + 1)
        plt.xticks([])
        plt.title(method.__name__)
        plt.plot(result, 'g-1')
        plt.plot(x_gt, y, 'r*')
        # plt.legend()

    plt.show()
'''


def eval_the(method, x, y):
    l_reg = method()
    l_reg.fit(x, y.ravel())
    # '''
    x_grid = data_grid()

    result = l_reg.predict(x_grid)

    x_gt = []
    for dd in x:
        no_gt = no_x(dd)
        x_gt.append(no_gt)

    # x_grid = []
    # for dd in data_grid:
        # no_gt = no_x(dd)
        # x_grid.append(no_gt)
    if True:
        plt.plot(result, 'g-1')
        plt.plot(x_gt, y, 'r*')
        plt.show()
    print(method.__name__ + "最优参数组合为: {}".format((x_grid[int(np.argmin(result))])))
    return result, x_gt, y


def data_grid():
    data = []
    nms = [120, 180, 240]
    fs = [15, 20, 25]
    rpms = [6000, 8000, 10000]
    mms = [20, 15, 10]

    for nm in nms:
        for f in fs:
            for rpm in rpms:
                for mm in mms:
                    data.append([nm, f, rpm, mm])

    return data


def read_data(path="data.txt"):
    """
    读取数据
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print("找不到数据文件")
        return 0

    data = np.loadtxt('data.txt', skiprows=1, encoding='utf-8')
    data = np.delete(data, -1, 1)
    print("共{}条数据".format(data.shape[0]))
    # 打乱数据
    # np.random.shuffle(data)
    # 十折交叉验证分为10组
    # data = data.reshape((10, -1, data.shape[1]))
    x = data[:, :4]
    y = data[:, 4:]
    return x, y


def no_x(x):
    # 复原顺序
    x = ((((x[0] / 60 - 2) * 3 + x[1] / 5 - 3) * 3 + x[2] / 2000 - 3) * 3 + 4 - x[3] / 5)
    # 特征重要性的排序
    # x = ((((4 - x[3] / 5) * 3 + x[0] / 60 - 2) * 3 + x[1] / 5 - 3) * 3 + x[2] / 2000 - 3)
    return x


# def get_data(data, i):
#    x_train, y_train, x_eval, y_eval =
#    x_eval = np.delete(data[i],-1, 1)


if __name__ == '__main__':
    run()
