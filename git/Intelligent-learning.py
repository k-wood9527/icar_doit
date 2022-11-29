import xgboost
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
import lightgbm  as lgb

def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax

data = pd.read_csv('icondata/2022.3.160.csv',encoding="unicode_escape")


X = data.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
x, x_min, x_max = feature_scalling(X)


for io in range(5):
    y = data.iloc[:, 10+io].values
    # x = MinMaxScaler().fit_transform(x)  # 归一化
    seed = 7
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=seed)  # 取0.3的数据作为测试数据
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # 进行5折交叉验证
    n_train = xtrain.shape[0]  # 统计训练样本数
    n_test = ytest.shape[0]  # 统计测试样本数
    # print(n_test)
    # print(n_train)
    models = [
              RandomForestRegressor(n_estimators=300, random_state=100),
              GradientBoostingRegressor(n_estimators=300, random_state=100),
              LGBMRegressor(max_depth=2,
                                  learning_rate=0.2883880652188603,
                                  n_estimators=190,
                                  num_leaves=29),
              RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 50]),
              LinearRegression(),
              SVR(kernel="linear"),
              SVR(kernel="rbf"),
              XGBRegressor(max_depth=2,
                                      learning_rate=0.13819987766133593,
                                      n_estimators=985,
                                      min_child_weight=9.542765428326693),
              ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=100)
    ]  # 这里选取了9个模型作为stacking的基学习器

    def get_oof(model, x_train, y_train, x_test):
        oof_train = np.zeros((n_train,))  # 构造一个1*4871的一维0矩阵
        oof_test = np.zeros((n_test,))  # 构造一个1*2088的一维0矩阵
        oof_test_skf = np.zeros((5, n_test))  # 5*2088

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            kf_x_train = x_train[train_index]  # 每一折训练3896个样本的X
            kf_y_train = y_train[train_index]  # 每一折训练3896个样本的Y
            kf_x_test = x_train[test_index]  # 每一折的1044测试样本的X
            model = model.fit(kf_x_train, kf_y_train)
            oof_train[test_index] = model.predict(kf_x_test)  # 每次产生974个预测值，最终5折后成为堆叠成为1*4871个训练样本的测试值
            oof_test_skf[i, :] = model.predict(
                x_test)  # 每次生成1*2088的测试集预测值，填oof_test_skf[i，：]，五次以后填满形成5*2088的预测值矩阵

        oof_test[:] = oof_test_skf.mean(axis=0)  # 把测试集的五次预测结果，求平均，形成一次预测结果
        return oof_train, oof_test, model  # 第一个返回值为第二层模型训练集的特征，1*4871；第二个返回值为第一层模型对测试集数据的预测1*2088，要作为第二层模型的训练集Xtest

    number_models = len(models)
    xtrain_new = np.zeros((n_train, number_models))
    xtest_new = np.zeros((n_test, number_models))  # 建立第二层的训练集和测试集

    for i, regressor in enumerate(models):
        xtrain_new[:, i], xtest_new[:, i], model = get_oof(regressor, xtrain, ytrain, xtest)

        # save_name = 'stacking' + str(io + 1) + str(i + 1) + '.pkl'
        # joblib.dump(model, './模型参数保存文件/' + save_name)  # 获得stacking的数据，对每个模型进行保存
    reg = LinearRegression()  # 第二层的学习器采用线性回归模型
    reg = reg.fit(xtrain_new, ytrain)

    y_test_predict = reg.predict(xtest_new)
    score = reg.score(xtest_new, ytest)
    rmse = np.sqrt(((y_test_predict - ytest) ** 2).mean())
    ave = np.mean(ytest)

    print('指标'+str(io))
    print('rmse: %.6f' % rmse)
    print('mae: %.6f' % mean_absolute_error(y_test_predict, ytest))
    print('r_square: %.6f' % r2_score(ytest, y_test_predict))

