# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:06:04 2020

@author: wangjingxian
"""
# 使用ID3算法进行分类
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import numpy as np
import array
from sklearn import model_selection

data = pd.read_csv('E:\data_mining\loudian_problem\data\dataset8_yuchuli.csv', encoding='utf-8')
#print (data.head(5))  # 查看数据

#检查缺失值并删除
print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isnull().sum())#返回每列包含的缺失值的个数
data.dropna(axis=0, how='any', inplace=True)


X = data.iloc[:,[2,3,7]]    
#X = data.iloc[:,[1,2]]
#X = data.iloc[:, 1:4] 
#X = data.iloc[:,[4]]
y = data.iloc[:, 8]
#print(len(y))
print(X.head(20))
print(y.head(20))


X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=0)

dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
dtc.fit(X_train, y_train)    # 训练模型
print ('输出训练准确率：', dtc.score(X_train,y_train))
print ('输出验证准确率：', dtc.score(X_test,y_test))


scores=model_selection.cross_val_score(dtc,X,y,cv=4)
scores=scores.mean()
print('k折交叉验证的平均准确率：',scores)


#使用未知数据进行结果预测
model=dtc.fit(X, y)
data_test=pd.read_csv('E:\data_mining\loudian_problem\data\dataset8_test.csv',encoding='utf-8')
#X_test=data_test.iloc[:,1:4]
#X_test= data_test.iloc[:,[1,2,3,4]]
X_test= data_test.iloc[:,[2,3,7]]
print(X_test)
#测试数据的预测结果
test_result=model.predict(X_test)
print(test_result)
#测试数据的真实结果
y_test=data_test.iloc[:,8]
print(y_test)
#print(len(y_test))
#测试数据的准确率
a=0
for i in range(len(y_test)):
    #print(y_test[i])
    if test_result[i]==y_test[i]:
        #print('true')
        a=a+1
    #else:
        #print('false')

accuracy=a/(len(y_test))
print('输出测试准确率：',accuracy)