import xgboost
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm  as lgb
from deepforest import CascadeForestRegressor


def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax

def fea(X,XMIN,XMAX):
    return X*(XMAX-XMIN)+XMIN
''
data = pd.read_csv('icondata/2022.3.160.csv',encoding="unicode_escape")


X = data.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
print(X[0])
X, x_min, x_max = feature_scalling(X)




def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax


class othermodel():
    def __init__(self):
        self.path = './'

    def LG_model(self):
        for i in range(1):
            Y = data.iloc[:, 14 + i].values
                # .reshape(-1,1)
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = lgb.LGBMRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('LGBM-指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))




            ##绘图
            y_true = y_test
            y_pred = y_pred
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # # 定义偏离程度大小
            # T = abs(y_true - y_pred) / y_true
            T = abs(y_true - y_pred)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            cm = plt.cm.get_cmap('rainbow_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=0.5, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Pred=True')
            plt.legend()
            plt.xlabel('True', size=15)
            plt.ylabel('Pred', size=15)
            plt.title('Tabnet_model')
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()

            # 绘制折线图
            plt.figure()
            plt.plot(np.arange(len(y_test[:300])), y_test[:300], 'r-', label='True', marker='.')

            plt.scatter(np.arange(len(y_pred[:300])), y_pred[:300], label='Predict', s=25, c='g')
            plt.title('Tabnet_model')
            plt.legend()  # 将样例显示出来
            plt.show()



    def XG_model(self):
        for i in range(1):
            Y = data.iloc[:, 14 + i].values
            print(Y)
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = xgboost.XGBRFRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            print(model.feature_importances_)

            plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
            plt.show()
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('XGBOOST指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))



            # ##绘图
            y_true = y_test
            y_pred = y_pred
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # # 定义偏离程度大小
            # T = abs(y_true - y_pred) / y_true
            T = abs(y_true - y_pred)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            cm = plt.cm.get_cmap('rainbow_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=0.5, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Pred=True')
            plt.legend()
            plt.xlabel('True', size=15)
            plt.ylabel('Pred', size=15)
            plt.title('XGBoost_model')
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()

            # 绘制折线图
            plt.figure()
            plt.plot(np.arange(len(y_test[:300])), y_test[:300], 'r-', label='True', marker='.')

            plt.scatter(np.arange(len(y_pred[:300])), y_pred[:300], label='Predict', s=25, c='g')
            plt.title('XGBoost_model')
            plt.legend()  # 将样例显示出来
            plt.show()


    def LR_model(self):
        for i in range(1):
            Y = data.iloc[:, 10 + i].values
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('LR-指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))



            # ##绘图
            y_true = y_test
            y_pred = y_pred
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # # 定义偏离程度大小
            # T = abs(y_true - y_pred) / y_true
            T = abs(y_true - y_pred)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            cm = plt.cm.get_cmap('rainbow_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=0.5, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Pred=True')
            plt.legend()
            plt.xlabel('True', size=15)
            plt.ylabel('Pred', size=15)
            plt.title('SVR_model')
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()

            # 绘制折线图
            plt.figure()
            plt.plot(np.arange(len(y_test[:300])), y_test[:300], 'r-', label='True', marker='.')

            plt.scatter(np.arange(len(y_pred[:300])), y_pred[:300], label='Predict', s=25, c='g')
            plt.title('SVR_model')
            plt.legend()  # 将样例显示出来
            plt.show()

    def KNN_model(self):
        for i in range(1):
            Y = data.iloc[:, 10 + i].values
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = KNeighborsRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('KNN-指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))



        ##绘图
            y_true = y_test
            y_pred = y_pred
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # # 定义偏离程度大小
            # T = abs(y_true - y_pred) / y_true
            T = abs(y_true - y_pred)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            cm = plt.cm.get_cmap('rainbow_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=0.5, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Pred=True')
            plt.legend()
            plt.xlabel('True', size=15)
            plt.ylabel('Pred', size=15)
            plt.title('SVR_model')
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()

            # 绘制折线图
            plt.figure()
            plt.plot(np.arange(len(y_test[:300])), y_test[:300], 'r-', label='True', marker='.')

            plt.scatter(np.arange(len(y_pred[:300])), y_pred[:300], label='Predict', s=25, c='g')
            plt.title('SVR_model')
            plt.legend()  # 将样例显示出来
            plt.show()

    def SVR_model(self):
        for i in range(1):
            Y = data.iloc[:, 10 + i].values
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = SVR()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('SVR-指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))


            ##绘图
            y_true = y_test
            y_pred = y_pred
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # # 定义偏离程度大小
            # T = abs(y_true - y_pred) / y_true
            T = abs(y_true - y_pred)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            cm = plt.cm.get_cmap('rainbow_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=0.5, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Pred=True')
            plt.legend()
            plt.xlabel('True', size=15)
            plt.ylabel('Pred', size=15)
            plt.title('KNN_model')
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()

            # 绘制折线图
            plt.figure()
            plt.plot(np.arange(len(y_test[:300])), y_test[:300], 'r-', label='True', marker='.')

            plt.scatter(np.arange(len(y_pred[:300])), y_pred[:300], label='Predict', s=25, c='g')
            plt.title('KNN_model')
            plt.legend()  # 将样例显示出来
            plt.show()




    def RF_model(self):
        for i in range(1):
            Y = data.iloc[:, 10 + i].values
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('RF-指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))




            # ##绘图
            y_true = y_test
            y_pred = y_pred
            max_flage = max(y_true.max(), y_pred.max())
            min_flage = min(y_true.min(), y_pred.min())
            step = (max_flage - min_flage) / 100
            x = range(int(max_flage))
            x = np.arange(min_flage, max_flage, step)
            y = x
            # plt.scatter(Y_true, y_result, s=45, label='predict')
            # # 定义偏离程度大小
            # T = abs(y_true - y_pred) / y_true
            T = abs(y_true - y_pred)
            plt.axes().set_facecolor('whitesmoke')
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            cm = plt.cm.get_cmap('rainbow_r')
            sc = plt.scatter(y_true, y_pred, s=5, label='Data Points', c=T, alpha=0.5, cmap=cm)
            plt.plot(x, y, c='#000000ff', linewidth=2, label='Pred=True')
            plt.legend()
            plt.xlabel('True', size=15)
            plt.ylabel('Pred', size=15)
            plt.title('LGBM_model')
            plt.xlim(min_flage, max_flage)
            plt.ylim(min_flage, max_flage)
            plt.grid(ls='--')
            plt.colorbar(sc)
            plt.show()

            # 绘制折线图
            plt.figure()
            plt.plot(np.arange(len(y_test[:300])), y_test[:300], 'r-', label='True', marker='.')

            plt.scatter(np.arange(len(y_pred[:300])), y_pred[:300], label='Predict', s=25, c='g')
            plt.title('LGBM_model')
            plt.legend()  # 将样例显示出来
            plt.show()
    def CF_model(self):
        for i in range(1):
            Y = data.iloc[:, 10 + i].values
            Y, y_min, y_max = feature_scalling(Y)
            seed = 7
            test_size = 0.30
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
            model = CascadeForestRegressor(n_bins=25,max_layers=30,n_estimators=1,n_trees=200,predictor="xgboost")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
            print('CF-指标' + str(i + 1))
            print('rmse: %.6f' % rmse)
            print('mae: %.6f' % mean_absolute_error(y_pred, y_test))
            print('r_square: %.6f' % r2_score(y_test, y_pred))

        #     T_pred = model.predict(T)
        #     # print(T_pred)
        #     # print(y_min,y_max)
        #     T_pred = fea(T_pred, y_min, y_max)
        #
        #     #     print(T_pred)
        #     for io in range(800):
        #         data_i[io, :] = [T_pred[io]]
        #     #
        #     #
        #     #     # save_name = 'xgb'+str(i+1)+'.pkl'
        #     #     # joblib.dump(model,'./模型参数保存文件/'+save_name)
        #     #
        # cols = ['1', '2', '3', '4', '5']
        # save_data = pd.DataFrame(data=data_i, columns=cols)
        # save_data.to_csv('验证结果集0323.csv', index=None)



other = othermodel()
other.LR_model()
other.KNN_model()
other.SVR_model()
other.RF_model()
other.XG_model()
other.LG_model()
other.CF_model()
