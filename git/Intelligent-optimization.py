#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Python code created by "Thieu Nguyen" at 21:29, 12/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

# Main paper (Please refer to the main paper):
# Slime Mould Algorithm: A New Method for Stochastic Optimization
# Shimin Li, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, Seyedali Mirjalili
# Future Generation Computer Systems,2020
# DOI: https://doi.org/10.1016/j.future.2020.03.055
# https://www.sciencedirect.com/science/article/pii/S0167739X19320941
# ------------------------------------------------------------------------------------------------------------
# Website of SMA: http://www.alimirjalili.com/SMA.html
# You can find and run the SMA code online at http://www.alimirjalili.com/SMA.html

# You can find the SMA paper at https://doi.org/10.1016/j.future.2020.03.055
# Please follow the paper for related updates in researchgate:
# https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization
# ------------------------------------------------------------------------------------------------------------
#  Main idea: Shimin Li
#  Author and programmer: Shimin Li,Ali Asghar Heidari,Huiling Chen
#  e-Mail: simonlishimin@foxmail.com
# ------------------------------------------------------------------------------------------------------------
#  Co-author:
#             Huiling Chen(chenhuiling.jlu@gmail.com)
#             Mingjing Wang(wangmingjing.style@gmail.com)
#             Ali Asghar Heidari(aliasghar68@gmail.com, as_heidari@ut.ac.ir)
#             Seyedali Mirjalili(ali.mirjalili@gmail.com)
#
#             Researchgate: Ali Asghar Heidari https://www.researchgate.net/profile/Ali_Asghar_Heidari
#             Researchgate: Seyedali Mirjalili https://www.researchgate.net/profile/Seyedali_Mirjalili
#             Researchgate: Huiling Chen https://www.researchgate.net/profile/Huiling_Chen
# ------------------------------------------------------------------------------------------------------------
from random import uniform

import joblib

from SMA import BaseSMA, OriginalSMA
from numpy import sum, pi, exp, sqrt, cos, mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from scipy.interpolate import make_interp_spline



## You can create whatever function you want here

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


model1 = joblib.load('icondata/axgb1.model')
model2 = joblib.load('icondata/axgb2.model')
model3 = joblib.load('icondata/axgb3.model')
model4 = joblib.load('icondata/axgb4.model')
model5 = joblib.load('icondata/axgb5.model')
# objective fuction
def fun_weight(X):
    '''
    :param X: decison variable
    :return: fitness value
    '''
    # x = np.array(X).reshape(1, -1)
    # fitness = model1.predict(x)
    # fitness = fitness[0]

    x = np.array(X).reshape(1, -1)
    fitness = model1.predict(x)
    fitness = fitness[0]


    torsion = model2.predict(x)
    xmin = 6082.07
    xmax = 11098.87
    torsion = torsion * (xmax - xmin) + xmin
    torsion = torsion[0]-10000
    # torsion = model1.predict(x)
    # xmin = 330.607
    # xmax = 378.052
    # torsion = torsion * (xmax - xmin) + xmin
    # torsion = 344-torsion[0]

    torsionk = model3.predict(x)
    xmin = 15.52534675
    xmax = 28.81635777
    torsionk = torsionk * (xmax - xmin) + xmin
    torsionk = torsionk[0]-27

    bend = model4.predict(x)
    xmin = 4948.02
    xmax = 8510.76
    bend = bend * (xmax - xmin) + xmin
    bend = bend[0]-6700
    # bend = model1.predict(x)
    # xmin = 330.607
    # xmax = 378.052
    # bend = bend * (xmax - xmin) + xmin
    # bend = 380-bend[0]

    mode = model5.predict(x)
    xmin = 29.05554634
    xmax = 40.54089146
    mode = mode * (xmax - xmin) + xmin
    mode=mode[0]-33


    if torsion > 0 and bend > 0 and torsionk > 0 and mode >0:
        xmin = 330.607
        xmax = 378.052

        # xmin = 4948.02
        # xmax = 8510.76

        fitness = fitness * (xmax - xmin) + xmin
        fitness=fitness

    else:
        fitness=uniform(20445,20450)

    return fitness


## You can create different bound for each dimension like this
# lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -100, -40, -50]
# ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 20, 200, 1000]
# problem_size = 18
## if you choose this way, the problem_size need to be same length as lb and ub

## Or bound is the same for all dimension like this
dim = 10
data = pd.read_csv('icondata/2022.3.160.csv',encoding="unicode_escape")










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
# lb = [-100]u
# ub = [100]
problem_size = 10
## if you choose this way, the problem_size can be anything you want


## Setting parameters
# obj_func = func_ackley
verbose = True
epoch = 200
pop_size = 72

md1 = BaseSMA(fun_weight, lb, ub, problem_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1,kb = md1.train()
# return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
# print(md1.train())
print(md1.solution[1])
print(best_pos1)
print(best_fit1)
# print(list_loss1)

# print('------------------------')
# print(kb[1])
# print(len(kb[1]))
# print('------------------------').

y=md1.solution[0]*8+1
# y=0.125*np.floor(y)
print(y)
y = np.array(y).reshape(1, -1)
fitness = model1.predict(y)
xmin = 330.607
xmax = 378.052
# xmin = 4948.02
# xmax = 8510.76

fitness = fitness * (xmax - xmin) + xmin

print(fitness)
xminone = 6082.07
xmaxone = 11098.87
# xminone = 330.607
# xmaxone = 378.052
torsion = model2.predict(y)
torsion = torsion * (xmaxone - xminone) + xminone
print('torsion %.4f'%torsion+'(min:9720.428529 max:10583.26693)'+' %.4f'%(100*(torsion-xminone)/xminone)+'%')

bend0 = model3.predict(y)
xminzero = 15.52534675
xmaxzero = 28.81635777
bend0 = bend0 * (xmaxzero - xminzero) + xminzero
print('bend %.4f'%bend0+'(min:6636.765412 max:7054.534234)'+' %.4f'%(100*(bend0-xminzero)/xminzero)+'%')

bend = model1.predict(y)
xmintwo = 330.607
xmaxtwo = 378.052
bend = bend * (xmaxtwo - xmintwo) + xmintwo
print('bend %.4f'%bend+'(min:6636.765412 max:7054.534234)'+' %.4f'%(100*(bend-xmintwo)/xmintwo)+'%')

mode = model5.predict(y)
xminthree = 29.05554634
xmaxthree = 40.54089146
mode = mode * (xmaxthree - xminthree) + xminthree
k=str((mode-xminthree)/xminthree)
print('mode %.4f'%mode+'(min:22.52195 max:25.89001)'+' %.4f'%(100*(mode-xminthree)/xminthree)+'%')

from PIL import Image
path="./spice/modelX.jpg"
imag=Image.open(path) #读取接头模型图片
img = imag.convert("RGBA")
y=md1.solution[0]*8+1
print(y[0])
a=[int(y[0]),int(y[1]),int(y[2]),int(y[3]),int(y[4]),int(y[5]),int(y[6]),int(y[7]),int(y[8]),int(y[9])] #优化结果输入
path1="./spice/A"+str(a[0])+"UP.JPG"
# path1=str(path1)
print(path1)
# icon=Image.open("./spice/A1UP.JPG")
icon=Image.open(path1)
icon=icon.resize((380,188))
imag.paste(icon,(455,206),mask=None)

icon=Image.open("./spice/00.png")
icon=icon.resize((230,208))
imag.paste(icon,(56,410),mask=None)

icon=Image.open("./spice/9_B_upr.PNG")
icon=icon.resize((380,188))
imag.paste(icon,(866,206),mask=None)

icon=Image.open("./spice/4_C_upr.PNG")
icon=icon.resize((380,188))
imag.paste(icon,(1340,206),mask=None)

icon=Image.open("./spice/1_D_upr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1732,206),mask=None)

icon=Image.open("./spice/2_C_mid.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1380,445),mask=None)


icon=Image.open("./spice/1_D_mid.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1805,445),mask=None)

icon=Image.open("./spice/1_A_lwr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(236,740),mask=None)

icon=Image.open("./spice/6_B_lwr.PNG")
icon=icon.resize((380,188))
imag.paste(icon,(802,740),mask=None)

icon=Image.open("./spice/1_C_lwr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1270,740),mask=None)

icon=Image.open("./spice/1_D_lwr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1800,740),mask=None)

imag.show() #生成优化拼接结果




# print(md1.loss_train)
#
# md2 = OriginalSMA(fun_weight, lb, ub, problem_size, verbose, epoch, pop_size)
# best_pos2, best_fit2, list_loss2 = md2.train()
# # return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
# print(best_pos2)
# print(best_fit2)
# print(list_loss2)
#
# y=md2.solution[0]
# y = np.array(y).reshape(1, -1)
# fitness = model1.predict(y)[0, 0]
# xminone = 9720.428529
# xmaxone = 10583.26693
# torsion = model2.predict(y)[0, 0]
# torsion = torsion * (xmaxone - xminone) + xminone
# print('torsion %.4f'%torsion)
# bend = model4.predict(y)[0, 0]
# xmintwo = 6636.765412
# xmaxtwo = 7054.534234999999
# bend = bend * (xmaxtwo - xmintwo) + xmintwo
# print('bend %.4f'%bend)
# mode = model5.predict(y)[0, 0]
# xminthree = 22.52195
# xmaxthree = 25.89001
# mode = mode * (xmaxthree - xminthree) + xminthree
# print('mode %.4f'%mode)



plt.rcParams['font.sans-serif']=['Times New Roman']


plt.plot(list_loss1)
plt.xlabel('SMA_tan')
plt.ylabel('Fitness')
plt.show()

plt.scatter(range(len(kb[1])),np.array(kb[1]),c="b",label='Initial population position')
plt.scatter(range(len(kb[100])),np.array(kb[100]),c="g",label='Mid population position')
plt.scatter(range(len(kb[199])),np.array(kb[199]),c="r",label='Final population position')
# plt.scatter(pso.record_value['Y'][99],300)
plt.xlabel('SMA_tan')
plt.ylabel('POP_Fitness')
plt.show()

h=0
j=[]
l=[]
while h<=199:

    j.append(mean(np.array(kb[h])))
    l.append(min(kb[h]))
    h=h+1
# print(j)
# print(l)
plt.plot(j)
plt.xlabel('SMA_tan')
plt.ylabel('AVG_Fitness')
plt.show()
plt.plot(l)
plt.xlabel('SMA_tan')
plt.ylabel('MIN_Fitness')
plt.show()
plt.plot(j)
plt.plot(l)
plt.xlabel('SMA_tan')
plt.ylabel('MIN_Fitness AND AVG_Fitness')
plt.show()










