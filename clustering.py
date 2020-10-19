# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:10:42 2020

@author: wangjingxian
"""


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
from numpy import array 
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import numpy as np
#import array
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mp1
import matplotlib.pyplot as plt


path='E:\data_mining\loudian_problem\data_hefei\dataset5.csv'
data=pd.read_csv(path)
data=data[-data.rtu1.isin([0])]
X=data.ix[:,7]
    
#scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
#X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则

#X_dataScale=X.transform(X.values.reshape(-1,1))#应用规则
    
#print(np.isnan(X_dataScale).any())
X_dataScale=pd.DataFrame(X)
X_dataScale.dropna(inplace=True)
    
print('使用的漏电数据条数为：',X_dataScale.shape[0])
#print(np.isnan(X_dataScale).any())

    
kmeans=KMeans(n_clusters=5).fit(X_dataScale)#构建并训练模型
    #print('聚类结果为：',kmeans.labels_)
    
quantity = pd.Series(kmeans.labels_).value_counts()
print( "聚类后每个类别的样本数量\n", (quantity))
print( "聚类后每个类别的样本数量\n", quantity[0],quantity[1],quantity[2],quantity[3],quantity[4])
    
    
sum=quantity[0]+quantity[1]+quantity[2]+quantity[3]+quantity[4]
scale0=quantity[0]/sum
scale1=quantity[1]/sum
scale2=quantity[2]/sum
scale3=quantity[3]/sum
scale4=quantity[4]/sum

print('每个类别所占比例为：\n',scale0,'\n',scale1,'\n',scale2,'\n',scale3,'\n',scale4)

'''
a=quantity[0]
b=quantity[1]
c=quantity[2]
'''
        
    
    
#获取聚类之后每个聚类中心的数据
resSeries = pd.Series(kmeans.labels_)
    
res0 = resSeries[resSeries.values == 0]
    #print("聚类后类别为0的数据\n",(data.iloc[res0.index]))
data0=data.iloc[res0.index]
data0_dianliu=data0.ix[:,7]
max0_dianliu=max(data0_dianliu)
min0_dianliu=min(data0_dianliu)
print('类别0的最小最大值为：',min0_dianliu,max0_dianliu)
    
res1 = resSeries[resSeries.values == 1]
#print("聚类后类别为1的数据\n",(data.iloc[res1.index]))
data1=data.iloc[res1.index]
data1_dianliu=data1.ix[:,7]
max1_dianliu=max(data1_dianliu)
min1_dianliu=min(data1_dianliu)
print('类别1的最大最小值为：',min1_dianliu,max1_dianliu)


res2 = resSeries[resSeries.values == 2]
#print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data2=data.iloc[res2.index]
data2_dianliu=data2.ix[:,7]
max2_dianliu=max(data2_dianliu)
min2_dianliu=min(data2_dianliu)
print('类别2的最大最小值为：',min2_dianliu,max2_dianliu)
    
res3 = resSeries[resSeries.values == 3]
#print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data3=data.iloc[res3.index]
data3_dianliu=data3.ix[:,7]
max3_dianliu=max(data3_dianliu)
min3_dianliu=min(data3_dianliu)
print('类别3的最大最小值为：',min3_dianliu,max3_dianliu)

  
res4 = resSeries[resSeries.values == 4]
#print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data4=data.iloc[res4.index]
data4_dianliu=data4.ix[:,7]
max4_dianliu=max(data4_dianliu)
min4_dianliu=min(data4_dianliu)
print('类别4的最大最小值为：',min4_dianliu,max4_dianliu)
 
'''   
res5 = resSeries[resSeries.values == 5]
#print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
data5=data.iloc[res5.index]
data5_dianliu=data5.ix[:,7]
max5_dianliu=max(data5_dianliu)
min5_dianliu=min(data5_dianliu)
print('类别5的最大最小值为：',min5_dianliu,max5_dianliu)
'''    
loudian1=[max0_dianliu,max1_dianliu,max2_dianliu,max3_dianliu,max4_dianliu]
loudian2=[min0_dianliu,min1_dianliu,min2_dianliu,min3_dianliu,min4_dianliu]

max_loudian=max(loudian1)
min_loudian=min(loudian2)
print('漏电电流的历史最小值为：',min_loudian)
print('漏电电流的历史最大值为：',max_loudian)

loudian1.sort()
loudian2.sort()
print('根据本条回路的历史漏电数据进行聚类建模分析，该条回路可分为三个不同等级：')
print('等级1：',loudian2[0],'--',loudian1[0])
print('等级2：',loudian2[1],'--',loudian1[1])
print('等级3：',loudian2[2],'--',loudian1[2])
print('等级4：',loudian2[3],'--',loudian1[3])
print('等级5：',loudian2[4],'--',loudian1[4])
'''
print('等级5：',loudian2[4],'--',loudian1[4])
print('等级6：',loudian2[5],'--',loudian1[5])
'''


#聚类结果评分：silhouette_score评分值（不需要真实值对比）,轮廓系数法
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
silhouetteScore=[]
for i in range(2,10):
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X.values.reshape(-1,1))#构建并训练模型
    kmeans=KMeans(n_clusters=i,random_state=123).fit(X_dataScale)#构建并训练模型
    #score=silhouette_score(X.values.reshape(-1,1),kmeans.labels_)
    score=silhouette_score(X_dataScale,kmeans.labels_)
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X)#构建并训练模型
    #score=silhouette_score(X,kmeans.labels_)
    silhouetteScore.append(score)
print('轮廓系数法不同分类树的准确率:\n',silhouetteScore)
#plt.figure(figsize=(10,6))
#plt.plot(range(2,15),silhouetteScore,linewidth=1.5,linestyle='-')
#plt.show()
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
for i in range(2,10):
    kmeans=KMeans(n_clusters=i,random_state=123).fit(X.values.reshape(-1,1))#构建并训练模型
    score=calinski_harabaz_score(X.values.reshape(-1,1),kmeans.labels_)
    #kmeans=KMeans(n_clusters=i,random_state=123).fit(X)#构建并训练模型
    #score=calinski_harabaz_score(X,kmeans.labels_)
    print('iris数据聚类数为%d类calinski_harabaz指数为:%f' %(i,score))






