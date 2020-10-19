# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:35:30 2020

@author: wangjingxian
"""

# 使用ID3算法进行分类
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import numpy as np
import array

data = pd.read_csv('E:\data_mining\data_classification_model\decision_tree_model\data\loudian_data_standard.csv', encoding='utf-8')
#print (data.head(5))  # 查看数据

#X = data.iloc[:, 1:5]    
#X = data.iloc[:,[1,2]]
#X = data.iloc[:, 1:4] 
X = data.iloc[:,[4]]
y = data.iloc[:, 8]
#print(len(y))
print(X.head(20))
print(y.head(20))


dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
dtc.fit(X, y)    # 训练模型
print ('输出训练准确率：', dtc.score(X,y))


#使用未知数据进行结果预测
model=dtc.fit(X, y)
data_test=pd.read_csv('E:\data_mining\data_classification_model\decision_tree_model\data\loudian_data_test.csv',encoding='utf-8')
#X_test=data_test.iloc[:,1:4]
#X_test= data_test.iloc[:,[1,2,3,4]]
X_test= data_test.iloc[:,[4]]
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
print('测试数据的准确率为：',accuracy)
        

'''
data_test = pd.read_csv('../data/test.csv', encoding='utf-8')
data_test.drop(['PassengerId'], axis=1, inplace=True)    # 舍弃ID列，不适合作为特征
# 数据是类别标签，将其转换为数，用1表示男，0表示女。
data_test.loc[data['Sex'] == 'male', 'Sex'] = 1
data_test.loc[data['Sex'] == 'female', 'Sex'] = 0
data_test.fillna(int(data.Age.mean()), inplace=True)
X_test = data_test.iloc[:, 1:3]    # 为便于展示，未考虑年龄（最后一列）
y = data.iloc[:, 0]
print(X_test)
test_result=model.predict(X_test)
print(test_result)
'''

# 可视化决策树，导出结果是一个dot文件，需要安装Graphviz才能转换为.pdf或.png格式
with open('./model/tree_loudian_shebei2_huilu1_wenshidu.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=X.columns, out_file=f)
