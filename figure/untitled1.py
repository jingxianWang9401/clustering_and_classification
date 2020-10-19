# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:11:56 2020

@author: wangjingxian
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False





fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

data = pd.read_csv("E:\\data_mining\\loudian_problem\\data_hefei\\dataset20_yuchuli2.csv",encoding="UTF-8")
#data=data[-data.rtu1.isin([0])]
data=data.ix[4000:,]
print(data.shape)


X=data['uptime_1600009']


Y0=data['rtu2']
#Y1=data['rtu2']
#Y2=data['rtu3']
#Y3=data['rtu4']
#plt.plot(X, Y0, color='purple', label='id1', linewidth=2.0)
#plt.plot(X, Y0, color='blue', label='id1', linewidth=1.5)

#plt.figure(figsize=(50,20))
plt.plot(X, Y0, color='red', label='rtu2', linewidth=1.5)


#plt.plot(X, Y1, color='blue', label='id2', linewidth=2.2)
#plt.plot(X, Y2, color='green', label='id3', linewidth=1.8)
#plt.plot(X, Y3, color='red', label='id4', linewidth=1.6)

plt.xticks(rotation=45)

#plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=1)
     
plt.margins(0)
#plt.subplots_adjust(bottom=0.10)
plt.subplots_adjust(bottom=0.20)
plt.xlabel('Time') #X轴标签
plt.ylabel("A") #Y轴标签

plt.show()

#plt.ylim(-10,60)    
    
    
#plt.savefig('E:\\data_mining\\loudian_problem\\figure\\data_figure\\dataset1.png',figsize=(80,40))