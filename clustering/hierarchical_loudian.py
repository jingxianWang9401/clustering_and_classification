# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:16:23 2020

@author: wangjingxian
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data=pd.read_csv('E:\data_mining\loudian_problem\data\dataset3.csv')

X=data.ix[:,7]

scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则

##设置分层聚类函数
linkages = ['ward', 'average', 'complete']
n_clusters_ = 3
ac = AgglomerativeClustering(linkage=linkages[2],n_clusters = n_clusters_)
##训练数据
ac.fit(X_dataScale)

##每个数据的分类
lables = ac.labels_
print(lables)

##簇中心的点的集合
#cluster_centers = ac.cluster_centers_
#print('cluster_centers:',cluster_centers)

##总共的标签分类
labels_unique = np.unique(lables)

##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

#print ("聚类中心\n", (ac.cluster_centers_))

quantity = pd.Series(ac.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(ac.labels_)
res0 = resSeries[resSeries.values == 0]
print("聚类后类别为0的数据\n",(data.iloc[res0.index]))
data0=data.iloc[res0.index]
data0_dianliu=data0.ix[:,7]
max0_dianliu=max(data0_dianliu)
min0_dianliu=min(data0_dianliu)
print('类别0的最大最小值为：\n',min0_dianliu,max0_dianliu)

res1 = resSeries[resSeries.values == 1]
print("聚类后类别为1的数据\n",(data.iloc[res1.index]))
data1=data.iloc[res1.index]
data1_dianliu=data1.ix[:,7]
max1_dianliu=max(data1_dianliu)
min1_dianliu=min(data1_dianliu)
print('类别1的最大最小值为：\n',min1_dianliu,max1_dianliu)

res2 = resSeries[resSeries.values == 2]
print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data2=data.iloc[res2.index]
data2_dianliu=data2.ix[:,7]
max2_dianliu=max(data2_dianliu)
min2_dianliu=min(data2_dianliu)
print('类别2的最大最小值为：\n',min2_dianliu,max2_dianliu)



res3 = resSeries[resSeries.values == 3]
print("聚类后类别为3的数据\n",(data.iloc[res3.index]))