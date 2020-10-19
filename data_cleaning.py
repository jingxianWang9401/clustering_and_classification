# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:03:13 2020

@author: wangjingxian
"""
import pandas as pd

#表格中含有特殊字符，将含有特殊字符的整行进行删除
datafile = u'E:/data_mining/loudian_problem/data/dataset4.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv


#检查缺失值并删除
print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isnull().sum())#返回每列包含的缺失值的个数
data.dropna(axis=0, how='any', inplace=True)



#df1=pd.DataFrame(data,columns=list('va'))
df1=pd.DataFrame(data)
data=df1.loc[ ~ data['1600003_rtu3'].isin([0])]#不加~是选取表格中含有1的行，加~是取反，删除表格中含有1的行
#data=df1.loc[ ~ data['1600003_rtu4'].isin([0])]#不加~是选取表格中含有1的行，加~是取反，删除表格中含有1的行
print('删除特殊字符后的数据为：',data)
data.to_csv('E:/data_mining/loudian_problem/data/dataset4_2.csv',index=False) #将数据重新写入excel