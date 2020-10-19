# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:55:00 2020

@author: wangjingxian
"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


data=pd.read_csv('E:\data_mining\loudian_problem\data\dataset3.csv')
X=data.ix[:,7]

scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则

##设置gmm函数
gmm = GaussianMixture(n_components=3, covariance_type='spherical').fit(X_dataScale)
##训练数据
y_pred = gmm.predict(X_dataScale)

print(y_pred)
'''
predicted_label=gmm.predict([[0.320347155,0.478602869]])
print('预测标签为：',predicted_label)
'''

##总共的标签分类
labels_unique = np.unique(y_pred)
##聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters聚类数量为 : %d" % n_clusters_)

#print ("聚类中心\n", (spectral_clustering.cluster_centers_))
quantity = pd.Series(y_pred).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类的数据
resSeries = pd.Series(y_pred)
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


#聚类结果评分：silhouette_score评分值（不需要真实值对比）,轮廓系数法
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
silhouetteScore=[]
for i in range(2,15):
    gmm=GaussianMixture(n_components=i,covariance_type='spherical').fit(X.values.reshape(-1,1))#构建并训练模型
    score=silhouette_score(X.values.reshape(-1,1),y_pred)
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X)#构建并训练模型
    #score=silhouette_score(X,kmeans.labels_)
    silhouetteScore.append(score)
print('轮廓系数法不同分类树的准确率:\n',silhouetteScore)
plt.figure(figsize=(10,6))
plt.plot(range(2,15),silhouetteScore,linewidth=1.5,linestyle='-')
plt.show()


#calinski_harabaz指数（不需要真实值对比）
from sklearn.metrics import calinski_harabaz_score
for i in range(2,7):
    gmm=GaussianMixture(n_components=i,covariance_type='spherical').fit(X.values.reshape(-1,1))#构建并训练模型
    score=calinski_harabaz_score(X.values.reshape(-1,1),y_pred)
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X)#构建并训练模型
    #score=calinski_harabaz_score(X,kmeans.labels_)
    print('iris数据聚类数为%d类calinski_harabaz指数为:%f' %(i,score))