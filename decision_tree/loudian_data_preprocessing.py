# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:38:24 2020

@author: wangjingxian
"""

import pandas as pd #导入pandas库

#显示完整数据
pd.set_option('display.max_columns',10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth',10000)

#读数据
inputfile = u'E:\data_mining\loudian_problem\data_hefei\dataset1.csv'
data= pd.read_csv(inputfile)

#天气编码，转化为算法可以处理的编码
data.loc[data['weather'] == '雨', 'weather'] = 0
data.loc[data['weather'] == '阴', 'weather'] = 1
data.loc[data['weather'] == '雾', 'weather'] = 2
data.loc[data['weather'] == '多云', 'weather'] = 3
data.loc[data['weather'] == '晴', 'weather'] = 4
data.loc[data['weather'] == '阵雨', 'weather'] = 0
print('天气编码：',data.head(15))


#风向编码，转化为算法可以处理的编码
data.loc[data['wind'] == '东风', 'wind'] = 0
data.loc[data['wind'] == '西风', 'wind'] = 1
data.loc[data['wind'] == '南风', 'wind'] = 2
data.loc[data['wind'] == '北风', 'wind'] = 3
data.loc[data['wind'] == '东南风', 'wind'] = 4
data.loc[data['wind'] == '东北风', 'wind'] = 5
data.loc[data['wind'] == '西南风', 'wind'] = 6
data.loc[data['wind'] == '西北风', 'wind'] = 7
data.loc[data['wind'] == '无持续风向', 'wind'] = 8
data.loc[data['wind'] == '风', 'wind'] = 9
print('风向编码：',data.head(15))



#漏电电流编码，<1000:0;1000<i<2000:1;i>2000:2
def current_coding(x):
    if x<42:
        return 0
    elif 42<x<103:
        return 1
    elif x>103:
        return 2
    
data['1600003_rtu1']= data['1600003_rtu1'].apply(lambda x: current_coding(x))

print('漏电电流编码：',data.head(20))
#data1.to_csv('E:/data_mining/data_classification_model/decision_tree_model/data/loudian_data1.csv')


#将温度、湿度和风力的单位去掉，并且转换为整形int数据
df=pd.DataFrame(data)


'''
df['temperature']=df['temperature'].apply(lambda x:x[0:-1])
df['humidity']=df['humidity'].apply(lambda x:x[0:-1])
df['winp']=df['winp'].apply(lambda x:x[0:-1])
'''


df['temperature']= df['temperature'].str.replace("℃","")
df['humidity']=df['humidity'].str.replace("%","")
df['winp']=df['winp'].str.replace("级","")


df.dropna(inplace=True)
#df= df.humidity.str.replace("%","")
#df= df.winp.str.replace("级","")


df[['temperature','humidity','winp']]=df[['temperature','humidity','winp']].astype('int') 
print('温度、适度、风力去单位并整数化：\n',df)

df.to_csv('E:/data_mining/loudian_problem/data_hefei/dataset1_yuchuli.csv')


