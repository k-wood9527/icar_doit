import xgboost
import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt
import seaborn as sns




data = pd.read_csv('icondata/2022.3.160.csv',encoding="unicode_escape")
#feature_name = data.columns.tolist()
feature_name = ['cm_rr_stamp_x', 'stamp_r_frt', 'stamp_r_rr', 'rail_otr_ctr_y', 'rail_lwr_y', 'stamp_ctr_y', 'stamp_otr_y', 'JXDC_dis_Z', 'MODE_FIRST', 'MODE_SECOND', 'STIFF_CX', 'STIFF_SG', 'STIFF_TOR', 'STIFF_WINGTIP', 'MASS']
print(feature_name)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data=data,columns=feature_name)
data.info()
X = data.iloc[:,[0,1,2,3,4,5,6,7,8,9]]

# scaler = MinMaxScaler()
def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax
# X = scaler.fit_transform(X)
# X,xmin,xmax = feature_scalling(X)


figure, ax = plt.subplots(figsize=(9, 12))

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 23,
        'style': 'italic'
         }

ax=sns.heatmap(X.corr(), square=True, annot=True,annot_kws={'size':7,'weight':'bold', 'color':'black','family': 'Times New Roman', }, ax=ax,cmap=sns.diverging_palette(20, 220, n=200),fmt=".2f")    #cmap=sns.diverging_palette(20, 220, n=200)
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)





plt.tick_params(labelsize=9,)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontstyle('italic') for label in labels]

# ax.set_xticklabels(size=1)


plt.show()



