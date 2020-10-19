# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:32:32 2020

@author: wangjingxian
"""


from sklearn.datasets import load_iris
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

#第一个功能：根据历史数据实现漏电风险提醒
def fengxiantixing(path):
    data=pd.read_csv(path)
    data=data[-data.rtu1.isin([0])]
    X=data.ix[:,7]
    
    scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
    X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则
    
    #print(np.isnan(X_dataScale).any())
    X_dataScale=pd.DataFrame(X_dataScale)
    X_dataScale.dropna(inplace=True)
    
    print('使用的漏电数据条数为：',X_dataScale.shape[0])
    #print(np.isnan(X_dataScale).any())

    
    kmeans=KMeans(n_clusters=3).fit(X_dataScale)#构建并训练模型
    #print('聚类结果为：',kmeans.labels_)
    
    quantity = pd.Series(kmeans.labels_).value_counts()
    print( "聚类后每个类别的样本数量\n", (quantity))
    
    print( "聚类后每个类别的样本数量\n", quantity[0],quantity[1],quantity[2])
    
    sum=quantity[0]+quantity[1]+quantity[2]
    scale0=quantity[0]/sum
    scale1=quantity[1]/sum
    scale2=quantity[2]/sum
    
    print('每个类别所占比例为：\n',scale0,'\n',scale1,'\n',scale2)
        
    
    
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
    
    loudian1=[max0_dianliu,max1_dianliu,max2_dianliu]
    loudian2=[min0_dianliu,min1_dianliu,min2_dianliu]
    max_loudian=max(loudian1)
    min_loudian=min(loudian2)
    print('漏电电流的历史最小值为：',min_loudian)
    print('漏电电流的历史最大值为：',max_loudian)
    
       
    
    
    if max_loudian<=1000:
        print('该线路历史最大漏电电流不超过1000mA，风险较低，但后续工况可能变化，请持续关注')
    if 1000<max_loudian<=2000:
        print('该线路历史最大电流介于1000mA到2000mA之间，有一定的风险，请密切关注，并注意该条线路的漏电是否和特殊天气相关')
    if min_loudian>2000:
        print('该条线路历史最小漏电电流高于2000mA，风险很高，请抓紧排查问题')
    if max_loudian>2000:
        print('该条线路历史最大漏电电流高于2000mA，风险较高，请抓紧排查问题')
    
    
    return loudian1,loudian2
    
   
'''
    # 生成数据
    labels = ['0', '1', '2']
    share = [a, b, c]

    # 设置分裂属性
    explode = [0, 0.1, 0]

    # 分裂饼图
    plt.pie(share, explode = explode,
        labels = labels, autopct = '%3.1f%%',
        startangle = 180, shadow = True,
        colors = ['c', 'r', 'g'])

    # 标题
    plt.title('loudian_a')
    plt.legend()
    plt.show()
'''
   
    
    
'''    
    max_loudian=max(loudian1)
    min_loudian=min(loudian2)
    print('漏电电流的历史最小值为：',min_loudian)
    print('漏电电流的历史最大值为：',max_loudian)
    
       
    
    
    if max_loudian<=1000:
        print('该线路历史最大漏电电流不超过1000mA，风险较低，但后续工况可能变化，请持续关注')
    if 1000<max_loudian<=2000:
        print('该线路历史最大电流介于1000mA到2000mA之间，有一定的风险，请密切关注，并注意该条线路的漏电是否和特殊天气相关')
    if min_loudian>2000:
        print('该条线路历史最小漏电电流高于2000mA，风险很高，请抓紧排查问题')
    if max_loudian>2000:
        print('该条线路历史最大漏电电流高于2000mA，风险较高，请抓紧排查问题')
    
    
    return loudian1,loudian2
'''

#第二个功能，根据聚类建模算法进行漏电等级划分
def loudiandengjihuafen(loudian1,loudian2):
    loudian1.sort()
    loudian2.sort()
    print('根据本条回路的历史漏电数据进行聚类建模分析，该条回路可分为三个不同等级：')
    print('低漏电等级1：',loudian2[0],'--',loudian1[0])
    print('中漏电等级2：',loudian2[1],'--',loudian1[1])
    print('高漏电等级3：',loudian2[2],'--',loudian1[2])
    #print('aaaaaaaaaaaaaaaaaaaaaa')
    #print('每个漏电等级的数量为：',quantity[1])
    '''
    # 设置输出文字类型
    mp1.rcParams['font.family'] = 'STFangsong'
    labels=['等级1','等级2','等级3']
    #quantity=array(quantity)
    loudian=[int(quantity[0]),int(quantity[1]),int(quantity[2])]
    #loudian=array(loudian)
    print(loudian)
    #fig=plt.figure()
    plt.pie(loudian,labels)
    plt.title('漏电统计')
    plt.show()
    plt.savefig('E:\data_mining\loudian_problem\loudiantongji.jpg')
    '''
    return loudian1,loudian2
    

#进行数据预处理为后续数据预测建模做准备 
def data_preprocessing(path1,path2,loudian1,loudian2):
    data= pd.read_csv(path1)
    data=data[-data.rtu1.isin([0])]
    #天气编码，转化为算法可以处理的编码
    data.loc[data['weather'] == '雨', 'weather'] = 0
    data.loc[data['weather'] == '阵雨', 'weather'] = 0
    
    data.loc[data['weather'] == '阴', 'weather'] = 1
    data.loc[data['weather'] == '雾', 'weather'] = 2
    data.loc[data['weather'] == '多云', 'weather'] = 3
    data.loc[data['weather'] == '晴', 'weather'] = 4
    
    data.loc[data['weather'] == '阵雪', 'weather'] = 5
    data.loc[data['weather'] == '雪', 'weather'] = 5
    data.loc[data['weather'] == '雨夹雪', 'weather'] = 6
    data.loc[data['weather'] == '扬沙', 'weather'] = 7
    
    #print('天气编码：',data.head(15))
    
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
    #print('风向编码：',data.head(15))
    
    def current_coding(x):
        '''
        if x<loudian1[0]:
            return 0
        elif loudian1[0]<x<loudian1[1]:
            return 1
        elif x>loudian1[1]:
            return 2
        '''
        if loudian2[0]<x<loudian1[0]:
            return 0
        elif loudian2[1]<x<loudian1[1]:
            return 1
        elif loudian2[2]<x<loudian1[2]:
            return 2
    
    data['rtu1']= data['rtu1'].apply(lambda x: current_coding(x))
    #print('漏电电流编码：',data.head(20))
    
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
    
    df[['temperature','humidity','winp']]=df[['temperature','humidity','winp']].astype('int') 
    #print('温度、适度、风力去单位并整数化：',df)
    
    df.to_csv(path2)


#第三个功能：进行预测建模，建立漏电电流和温度、湿度、空气质量、风力、风向、天气之间的关系。
def loudian_decision_tree(path_train,path_test,loudian1,loudian2):
    data = pd.read_csv(path_train, encoding='utf-8')
    
    #检查缺失值并删除
    #print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isnull().sum())#返回每列包含的缺失值的个数
    data.dropna(axis=0, how='any', inplace=True)
    
    X = data.iloc[:,[2,3,7]]    
    y = data.iloc[:, 8]
    #print(X.head(20))
    #print(y.head(20))
    
    X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
    
    dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
    dtc.fit(X_train, y_train)    # 训练模型
    
    train_accuracy=dtc.score(X_train,y_train)
    validation_accuracy=dtc.score(X_test,y_test)
    print ('输出训练准确率：', train_accuracy)
    print ('输出验证准确率：', validation_accuracy)
    
    scores_cross_validation=model_selection.cross_val_score(dtc,X,y,cv=4)
    scores_mean=scores_cross_validation.mean()
    print('k折交叉验证的平均准确率：',scores_mean)
    if validation_accuracy<0.6:
        print('该线路漏电情况和天气等因素无关，相关性为%.2f%%,请关注其他因素产生的漏电增加原因'%(validation_accuracy*100))
    elif 0.6<=validation_accuracy<0.8:
        print('该线路漏电电流值和天气、温湿度等因素有关系，但是关系不密切,相关性为%.2f%%'%(validation_accuracy*100))
    elif validation_accuracy>=0.8:
        print('该线路漏电电流值和天气、温湿度等因素关系密切,相关性为%.2f%%'%(validation_accuracy*100))
        
        
'''
        #使用未知数据进行结果预测
        model=dtc.fit(X, y)
        data_test=pd.read_csv(path_test)
        X_test= data_test.iloc[:,[2,3,7]]
        time_weather=data_test.iloc[:,[1,2,3,7]]
        
        time_weather.loc[time_weather['weather'] == 0, 'weather'] = '雨'
        time_weather.loc[time_weather['weather'] == 1, 'weather'] = '阴'
        time_weather.loc[time_weather['weather'] == 2, 'weather'] = '雾'
        time_weather.loc[time_weather['weather'] == 3, 'weather'] = '多云'
        time_weather.loc[time_weather['weather'] == 4, 'weather'] = '晴'
        
        #将温度、湿度和风力的单位去掉，并且转换为整形int数据
        df=pd.DataFrame(time_weather)
        df[['temperature','humidity']]=df[['temperature','humidity']].astype('str') 
        df['temperature']=df['temperature'].apply(lambda x:x+'℃')
        df['humidity']=df['humidity'].apply(lambda x:x+'%')
       
        print('明天的时间点对应的天气预报（包括：温度、湿度、天气）情况为：',time_weather.head(20))
        #测试数据的预测结果
        test_result=model.predict(X_test)
        #predictions_proba=alg.predict_proba(X_test)        
        #predictions_proba1=predictions_proba.max(axis=1)
        real_result=test_result+1
        #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的漏电等级为：',real_result)
        for i in range(len(test_result)):
            if test_result[i]==0:
                real_result[i]=test_result[i]+1
                print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第%d条数据漏电等级为：%d'%(i,real_result[i]))
                real_time=time_weather.iloc[i]
                print('时间和天气预报为：',real_time)
                print('漏电电流大致范围是：',loudian2[0],'--',loudian1[0])
            if test_result[i]==1:
                real_result[i]=test_result[i]+1
                print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.iloc[i]
                print('时间和天气预报为：',real_time)
                print('漏电电流大致范围是：',loudian2[1],'--',loudian1[1])
            if test_result[i]==2:
                real_result[i]=test_result[i]+1
                print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.iloc[i]
                print('时间和天气预报为：',real_time)
                print('漏电电流大致范围是：',loudian2[2],'--',loudian1[2])
'''

        
    
path1='E:\data_mining\loudian_problem\data_hefei\dataset44.csv'

path2='E:/data_mining/loudian_problem/data_hefei/dataset44_yuchuli.csv'

path_test='E:\data_mining\loudian_problem\data_hefei\dataset1_test.csv'

loudian1,loudian2=fengxiantixing(path1)

loudian1,loudian2=loudiandengjihuafen(loudian1,loudian2)

#print('aaaaaaaaaaaaaaaaaaaaaa')

data_preprocessing(path1,path2,loudian1,loudian2)

loudian_decision_tree(path2,path_test,loudian1,loudian2)
    
    
