
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
# 1.决策树回归 ExtraTree极端随机数回归
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
# 2.线性回归
from sklearn.linear_model import LinearRegression
# 3.SVM回归
from sklearn.svm import SVR
# 4.kNN回归
from sklearn.neighbors import KNeighborsRegressor
# 5.随机森林回归 Adaboost回归 GBRT回归 Bagging回归
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor


def run():
    data, gt = read_data()
    '''
    methods = (LinearRegression, DecisionTreeRegressor, KNeighborsRegressor, SVR, ExtraTreeRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor)
    # methods = (SVR, KNeighborsRegressor)
    for method in methods:
        reg = method()
        cv=ShuffleSplit(n_splits=10,test_size=0.3,random_state=0)
        scores = cross_val_score(reg, data, gt.ravel(), cv=cv)
        print(method.__name__, scores)

    '''
    # 线性回归
    l_reg = LinearRegression()# n_estimators=50)
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    l_scores = cross_val_score(l_reg, data, gt.ravel(), cv=cv, scoring="r2")
    print(l_scores)
    l_reg.fit(data, gt)

    data_grid = []
    nms = [120, 180, 240]
    fs = [15, 20, 25]
    rpms = [6000, 8000, 10000]
    mms = [10, 15, 20]
    for nm in nms:
        for f in fs:
            for rpm in rpms:
                for mm in mms:
                    data_grid.append([nm, f, rpm, mm])

    result = l_reg.predict(data_grid)
    plt.plot(result, 'g-s')
    x_gt = []
    for dd in data:
        no_gt = ((((dd[0]/60 - 2)*3 + dd[1]/5 -3)*3 + dd[2]/2000 - 3)*3 + dd[3]/5 - 2)
        x_gt.append(no_gt)

    plt.plot(x_gt, gt)
    plt.show()
    print(data_grid[np.argmin(result)])

    '''
    t_reg = DecisionTreeRegressor()
    t_scores = cross_val_score(t_reg, data, gt, cv=10, scoring="r2")
    print(t_scores)
    '''


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


# def get_data(data, i):
#    x_train, y_train, x_eval, y_eval =
#    x_eval = np.delete(data[i],-1, 1)


if __name__ == '__main__':
    run()
