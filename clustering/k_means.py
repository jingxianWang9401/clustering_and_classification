# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:02:09 2020

@author: wangjingxian
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
from numpy import array 

data=pd.read_csv('E:\data_mining\loudian_problem\data\dataset3.csv')
X=data.ix[:,7]

'''
iris=load_iris()
iris_data=iris['data']#数据特征
iris_target=iris['target']#标签
iris_names=iris['feature_names']#特征对应的分类名称
'''

scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则

#scale=MinMaxScaler().fit(X)#训练规则
#X_dataScale=scale.transform(X)#应用规则


kmeans=KMeans(n_clusters=3).fit(X_dataScale)#构建并训练模型
#kmeans=KMeans(n_clusters=3,random_state=123).fit(X)#构建并训练模型

#print('构建的KMeans模型为：',kmeans)
print('聚类结果为：',kmeans.labels_)
#label = kmeans.labels_
#print(label)


#print ("聚类中心\n", (dbscan.cluster_centers_))

quantity = pd.Series(kmeans.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))

#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(kmeans.labels_)


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



'''
res3 = resSeries[resSeries.values == 3]
print("聚类后类别为3的数据\n",(data.iloc[res2.index]))
data3=data.iloc[res3.index]
data3_dianliu=data3.ix[:,5]
max3_dianliu=max(data3_dianliu)
min3_dianliu=min(data3_dianliu)
print('类别3的最大最小值为：\n',min3_dianliu,max3_dianliu)


res4 = resSeries[resSeries.values == 4]
print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data4=data.iloc[res4.index]
data4_dianliu=data4.ix[:,5]
max4_dianliu=max(data4_dianliu)
min4_dianliu=min(data4_dianliu)
print('类别4的最大最小值为：\n',min4_dianliu,max4_dianliu)
'''

#predict_result=kmeans.predict([[0.320347155,0.478602869]])
#print('预测类别为：',predict_result[0])


'''
#聚类结果评分:FMI评价分值（需要真实值）
from sklearn.metrics import fowlkes_mallows_score
for i in range(2,7):
    kmeans=KMeans(n_clusters=i,random_state=123).fit(iris_data)#构建并训练模型
    score=fowlkes_mallows_score(iris_target,kmeans.labels_)
    print('iris数据集聚类为%d类FMI评价分值为：%f' %(i,score))
'''    
      
    
'''
iris数据集聚类为2类FMI评价分值为：0.750473
iris数据集聚类为3类FMI评价分值为：0.820808
iris数据集聚类为4类FMI评价分值为：0.753970
iris数据集聚类为5类FMI评价分值为：0.725483
iris数据集聚类为6类FMI评价分值为：0.614345
'''
    
#聚类结果评分：silhouette_score评分值（不需要真实值对比）,轮廓系数法
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
silhouetteScore=[]
for i in range(2,8):
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X.values.reshape(-1,1))#构建并训练模型
    kmeans=KMeans(n_clusters=i,random_state=123).fit(X_dataScale)#构建并训练模型
    #score=silhouette_score(X.values.reshape(-1,1),kmeans.labels_)
    score=silhouette_score(X_dataScale,kmeans.labels_)
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X)#构建并训练模型
    #score=silhouette_score(X,kmeans.labels_)
    silhouetteScore.append(score)
print('轮廓系数法不同分类树的准确率:\n',silhouetteScore)
plt.figure(figsize=(10,6))
plt.plot(range(2,15),silhouetteScore,linewidth=1.5,linestyle='-')
plt.show()
'''
适用于实际类别信息未知情况下，对于单个样本，设：
a是与它同类别中其他样本的平均距离，
b是与它距离最近不同类别中样本的平均距离，轮廓系数为：
S=（b-a)/max(a,b)
对于一个样本集合，它的轮廓系数是所有样本轮廓系数的平均值
轮廓系数取值范围是[-1,1],同类别样本越距离相近且不同类别样本距离越远，分数越高
'''


#calinski_harabaz指数（不需要真实值对比）
from sklearn.metrics import calinski_harabaz_score
for i in range(2,8):
    kmeans=KMeans(n_clusters=i,random_state=123).fit(X.values.reshape(-1,1))#构建并训练模型
    score=calinski_harabaz_score(X.values.reshape(-1,1),kmeans.labels_)
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X)#构建并训练模型
    #score=calinski_harabaz_score(X,kmeans.labels_)
    print('iris数据聚类数为%d类calinski_harabaz指数为:%f' %(i,score))
    
'''
iris数据聚类数为2类calinski_harabaz指数为:513.924546
iris数据聚类数为3类calinski_harabaz指数为:561.627757
iris数据聚类数为4类calinski_harabaz指数为:530.487142
iris数据聚类数为5类calinski_harabaz指数为:495.541488
iris数据聚类数为6类calinski_harabaz指数为:469.836633

得到的Calinski-Harabasz分数值ss越大则聚类效果越好。
也就是说，类别内部数据的协方差越小越好，
类别之间的协方差越大越好，
这样的Calinski-Harabasz分数会高。 
在scikit-learn中， Calinski-Harabasz Index对应的方法是metrics.calinski_harabaz_score. 
在真实的分群label不知道的情况下，可以作为评估模型的一个指标。 
同时，数值越小可以理解为：组间协方差很小，组与组之间界限不明显。 
与轮廓系数的对比，笔者觉得最大的优势：快！相差几百倍！毫秒级
'''   