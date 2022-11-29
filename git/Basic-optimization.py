# -*- enconding: utf-8 -*-
# @ModuleName: optimization
# @Function:  optimize by tabnet
# @Author: Yanzhan Chen
# @Time: 2021/10/22 8:38



import pandas as pd
from numpy.ma import mean
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import copy
from sko.PSO import PSO

def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax

# variable range
def cal_range(data):
    '''
    :param data: type of DataFrame
    :return:  max list and min list
    '''
    up_limit = data.max().to_list()
    down_limit = data.min().to_list()
    return up_limit,down_limit


model1=joblib.load('icondata/axgb1.model')
model2=joblib.load('icondata/axgb2.model')
model3=joblib.load('icondata/axgb3.model')
model4=joblib.load('icondata/axgb4.model')
model5=joblib.load('icondata/axgb5.model')
# objective fuction
def fun(X,xmin=330.607,xmax=378.052):
    '''
    :param X: decison variable
    :return: fitness value
    '''
    x = np.array(X).reshape(1,-1)
    fitness = model1.predict(x)
    fitness = fitness*(xmax-xmin)+xmin
    fitness = fitness[0]
    return fitness

def fun_pso(X):
    '''
    :param X: decison variable
    :return: fitness value
    '''

    x = np.array(X).reshape(1, -1)
    fitness = model1.predict(x)
    xmin = 330.607
    xmax = 378.052
    fitness = fitness * (xmax - xmin) + xmin
    fitness = fitness[0]
    return fitness

def obj_funone(X):
    x = np.array(X).reshape(1, -1)
    torsion = model2.predict(x)
    xmin = 6082.07
    xmax = 11098.87
    torsion = torsion * (xmax - xmin) + xmin
    torsion=torsion[0]
    return 10000 - torsion

def obj_funtwo(X):
    x = np.array(X).reshape(1, -1)
    torsionk = model3.predict(x)
    xmin = 15.52534675
    xmax = 28.81635777
    torsionk = torsionk * (xmax - xmin) + xmin
    torsionk = torsionk[0]
    return 28-torsionk



def obj_funthree(X):
    x = np.array(X).reshape(1, -1)
    bend = model4.predict(x)
    xmin = 4948.02
    xmax = 8510.76
    bend = bend * (xmax - xmin) + xmin
    bend=bend[0]
    return 6900 - bend
def obj_funfour(X):
    x = np.array(X).reshape(1, -1)
    mode = model5.predict(x)
    xmin = 29.05554634
    xmax = 40.54089146
    mode = mode * (xmax - xmin) + xmin
    mode=mode[0]
    return 35-mode
# parameters
dim = 10
data = pd.read_csv('icondata/022.3.160.csv',encoding="unicode_escape")

X = data.iloc[:,0:14].values
X, x_min, x_max = feature_scalling(X)

ub,lb = cal_range(data)
UB = copy.deepcopy(ub)
LB = copy.deepcopy(lb)

ub = np.array(ub)
lb = np.array((lb))
ub1 = (ub-0.01) .tolist()
lb1 =(lb-1) .tolist()

ub = ub1[0:dim]
lb = lb1[0:dim]
print(ub)
print(lb)
print('*'*20)

# gwo_hist,obj1_hist,obj2_hist = gwo(obj_func=fun,obj_funone=obj_funone,obj_funtwo=obj_funtwo,limit1=torsion_limit,limit2=bend_limit,
#             SearchAgents=800,Max_iteration=100,dim=dim,lb=lb,ub=ub,
#                                    norm_min=[LB[720],LB[721],LB[722]],
#                                    norm_max=[UB[720],UB[721],UB[722]])
#
#
# plt.plot(gwo_hist)
# plt.show()


constraint_ueq = (
    obj_funone,
    obj_funtwo,
    obj_funthree,
    obj_funfour
)

pso = PSO(func=fun_pso, n_dim=10, pop=72, max_iter=200, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5,constraint_ueq=constraint_ueq)
pso.record_mode=True
pso.run()
print( 'best_y is', pso.gbest_y)

t=pso.best_x
print(t)
# print( 'best_y is', pso.gbest_y)
def pso_t(y):
    y = np.array(y).reshape(1, -1)
    fitness = model1.predict(y)
    xmin = 330.607
    xmax = 378.052

    fitness = fitness * (xmax - xmin) + xmin
    print(fitness)
    xminone = 6082.07
    xmaxone = 11098.87
    torsion = model2.predict(y)
    torsion = torsion * (xmaxone - xminone) + xminone
    print('torsion %.4f' % torsion + '(min:9720.428529 max:10583.26693)' + ' %.4f' % (
                100 * (torsion - xminone) / xminone) + '%')

    bend0 = model3.predict(y)
    xminzero = 15.52534675
    xmaxzero = 28.81635777
    bend0 = bend0 * (xmaxzero - xminzero) + xminzero
    print('bend %.4f' % bend0 + '(min:6636.765412 max:7054.534234)' + ' %.4f' % (
                100 * (bend0 - xminzero) / xminzero) + '%')

    bend = model4.predict(y)
    xmintwo = 4948.02
    xmaxtwo = 8510.76
    bend = bend * (xmaxtwo - xmintwo) + xmintwo
    print('bend %.4f' % bend + '(min:6636.765412 max:7054.534234)' + ' %.4f' % (100 * (bend - xmintwo) / xmintwo) + '%')

    mode = model5.predict(y)
    xminthree = 29.05554634
    xmaxthree = 40.54089146
    mode = mode * (xmaxthree - xminthree) + xminthree
    k = str((mode - xminthree) / xminthree)
    print('mode %.4f' % mode + '(min:22.52195 max:25.89001)' + ' %.4f' % (100 * (mode - xminthree) / xminthree) + '%')

    return fitness
pso_t(t)


plt.rcParams['font.sans-serif']=['Times New Roman']
plt.plot(pso.gbest_y_hist)
plt.xlabel('PSO')
plt.ylabel('Fitness')
plt.show()
list2=pso.record_value['Y']
print(list2)

plt.rcParams['font.sans-serif']=['Times New Roman']
# print(pso.all_history_Y)

plt.plot(pso.gbest_y_hist)
# plt.plot(pso.gbest_y_hist)
plt.xlabel('PSO')
plt.ylabel('Fitness')
plt.show()

print(pso.record_value['Y'])
print(pso.gbest_y_hist)

# print('pso_recors_mode:',pso.record_value['Y'][1])
list2=pso.record_value['Y']
print(list2)
k=mean(np.array(pso.record_value['Y'][1]).reshape(1, -1))
oa=0
while oa<=199:
    tt_list=pso.record_value['Y'][oa]
    for oi in range(len(tt_list)):
        if tt_list[oi]>400:
            tt_list[oi]=380
    oa = oa + 1

plt.plot(range(len(pso.record_value['Y'][1])),pso.record_value['Y'][1],c="b",label='Initial population position')
plt.plot(range(len(pso.record_value['Y'][10])),pso.record_value['Y'][10],c="g",label='Mid population position')
plt.plot(range(len(pso.record_value['Y'][199])),pso.record_value['Y'][199],c="r",label='Final population position')

# plt.scatter(pso.record_value['Y'][99],300)
plt.xlabel('PSO')
plt.ylabel('POP_Fitness')
plt.show()
k=0
j=[]
l=[]
while k<=199:
    j.append(mean(np.array(pso.record_value['Y'][k]).reshape(1, -1)))
    l.append(min(pso.record_value['Y'][k]))
    k=k+1
# print(j)
# print(l)
plt.plot(j)
plt.xlabel('PSO')
plt.ylabel('AVG_Fitness')
plt.show()
plt.plot(l)
plt.xlabel('PSO')
plt.ylabel('MIN_Fitness')
plt.show()
plt.plot(j)
plt.plot(l)
plt.xlabel('PSO')
plt.ylabel('MIN_Fitness AND AVG_Fitness')
plt.show()









