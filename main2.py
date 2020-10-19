# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:32:56 2020

@author: wangjingxian
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
from numpy import array 
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mp1
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


#第一个功能：根据历史数据实现漏电风险提醒
def fengxiantixing(path):
    data=pd.read_csv(path)
    data=data[-data.rtu4.isin([0])]#删除漏电数据为0的所有行
    X=data.ix[:,7]
    X=X.values.reshape(-1,1)
    
    #以下为数据归一化过程，该问题自变量数据级别差距不大，暂时不需要
    #scale=MinMaxScaler().fit(X.values.reshape(-1,1))#训练规则
    #X_dataScale=scale.transform(X.values.reshape(-1,1))#应用规则    
    #print(np.isnan(X_dataScale).any())
    #X_dataScale=pd.DataFrame(X_dataScale)
    #X_dataScale.dropna(how='any',inplace=True)    
    #print('除去漏电电流为0的数据使用的漏电数据条数为：',X_dataScale.shape[0])
    
    print('除去漏电电流为0的数据，数据集使用的漏电数据条数为：',X.shape[0])
    
    
    if X.shape[0]<1000:
        print('该条线路可用的有效数据量过少，低于1000条数据，参考性较小')
    #print(np.isnan(X_dataScale).any())

    
    #kmeans=KMeans(n_clusters=5).fit(X_dataScale)#构建并训练模型
    kmeans=KMeans(n_clusters=5).fit(X)#构建并训练模型
    #print('聚类结果为：',kmeans.labels_)
    
    quantity = pd.Series(kmeans.labels_).value_counts()
    print( "聚类后每个类别的样本数量\n", (quantity))
    #print( "聚类后每个类别的样本数量\n", quantity[0],quantity[1],quantity[2],quantity[3],quantity[4])
    
    sum=quantity[0]+quantity[1]+quantity[2]+quantity[3]+quantity[4]
    scale0=quantity[0]/sum
    scale1=quantity[1]/sum
    scale2=quantity[2]/sum
    scale3=quantity[3]/sum
    scale4=quantity[4]/sum
    print('每个聚类类别所占数据集总量的比例为：\n',scale0,'\n',scale1,'\n',scale2,'\n', scale3,'\n',scale4)
        
    scale=[scale0,scale1,scale2,scale3,scale4]
    max_scale=max(scale)
    division0=max_scale/scale0
    division1=max_scale/scale1
    division2=max_scale/scale2
    division3=max_scale/scale3
    division4=max_scale/scale4
    
    
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
    
    
    loudian1=[max0_dianliu,max1_dianliu,max2_dianliu,max3_dianliu,max4_dianliu]
    loudian2=[min0_dianliu,min1_dianliu,min2_dianliu,min3_dianliu,min4_dianliu]
       

        # 生成数据
    labels=['漏电电流：%d--%d'%(min0_dianliu,max0_dianliu),'漏电电流：%d--%d'%(min1_dianliu,max1_dianliu),'漏电电流：%d--%d'%(min2_dianliu,max2_dianliu),'漏电电流：%d--%d'%(min3_dianliu,max3_dianliu),'漏电电流：%d--%d'%(min4_dianliu,max4_dianliu)]
    share = [scale0, scale1, scale2,scale3,scale4]
    
    #解决汉字乱码问题
    matplotlib.rcParams['font.sans-serif']=['SimHei']  #使用指定的汉字字体类型（此处为黑体）
    plt.pie(share,labels = labels,autopct='%1.2f%%')
    #plt.pie(y_series,colors=colors,explode=explode,labels=y_series.index,shadow=True,textprops={'fontsize': 12, 'color': 'black'},autopct='%1.1f%%',pctdistance = 0.8)
    plt.title('漏电电流数据离散自动聚类')# 标题
    plt.legend()
    plt.show()
       
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
    
    #将数据量最多的等级占比/每个等级数据量占比>15的等级数据舍去，最简单的异常数据丢弃。
    class_num=5
    df=pd.DataFrame(data)
    if division0>15:
        #df = df.drop(df[min0_dianliu<df.rtu1.all() < max0_dianliu].index)
        df = df.drop(df[(df.rtu4 <=max0_dianliu) & (df.rtu4 >= min0_dianliu)].index)
        class_num=class_num-1
        
    if division1>15:
        #df = df.drop(df[min1_dianliu<df.rtu1.all() < max1_dianliu].index)
        df = df.drop(df[(df.rtu4 <= max1_dianliu) & (df.rtu4 >= min1_dianliu)].index)
        class_num=class_num-1

    if division2>15:
        #df = df.drop(df[min2_dianliu<df.rtu1.all() < max2_dianliu].index)
        df = df.drop(df[(df.rtu4 <= max2_dianliu) & (df.rtu4 >= min2_dianliu)].index)
        class_num=class_num-1

    if division3>15:
        #df = df.drop(df[min3_dianliu<df.rtu1.all()< max3_dianliu].index)
        df = df.drop(df[(df.rtu4 <= max3_dianliu) & (df.rtu4 >= min3_dianliu)].index)
        class_num=class_num-1

    if division4>15:
        #df = df.drop(df[min4_dianliu<df.rtu1.all()< max4_dianliu].index)
        df = df.drop(df[(df.rtu4 <= max4_dianliu) & (df.rtu4 >= min4_dianliu)].index)
        class_num=class_num-1

    
    print('最后使用的漏电分类数为：',class_num)
    if class_num<3:
        print('该线路异常数据点过多，不宜进行和天气的关联和预测，建模的参考性较小。')
        
    data=df
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
    
    df=data
    df['temperature']= df['temperature'].str.replace("℃","")
    df['humidity']=df['humidity'].str.replace("%","")
    df['winp']=df['winp'].str.replace("级","")
    
    df.dropna(inplace=True)
    df[['temperature','humidity','winp']]=df[['temperature','humidity','winp']].astype('int') 
    df.to_csv(path2)    
    return loudian1,loudian2


#第二个功能，根据聚类建模算法进行漏电等级划分
def loudiandengjihuafen(loudian1,loudian2):
    loudian1.sort()
    loudian2.sort()
    print('根据本条回路的历史漏电数据进行聚类建模分析，该条回路可分为五个不同等级：')
    print('等级1：',loudian2[0],'--',loudian1[0])
    print('等级2：',loudian2[1],'--',loudian1[1])
    print('等级3：',loudian2[2],'--',loudian1[2])
    print('等级4：',loudian2[3],'--',loudian1[3])
    print('等级5：',loudian2[4],'--',loudian1[4])
    return loudian1,loudian2
    

#进行数据预处理（漏电数据进行打标签）为后续数据预测建模做准备 
def data_preprocessing(path2,path3,loudian1,loudian2):
    data= pd.read_csv(path2)
    #print(loudian1)
    #print(loudian2)
    def current_coding(x):
        if loudian2[0]<=x<=loudian1[0]:
            #print('bbbbb')
            return 0
        if loudian2[1]<=x<=loudian1[1]:
            #print('eeeeeeee')            
            return 1
        if loudian2[2]<=x<=loudian1[2]:
            #print('ddddd')
            return 2
        if loudian2[3]<=x<=loudian1[3]:
            #print('ccccccccc')
            return 3
        if loudian2[4]<=x<=loudian1[4]:
            #print('aaaaaaa')
            return 4

    data['rtu4']= data['rtu4'].apply(lambda x: current_coding(x))

    df=pd.DataFrame(data)
    df.to_csv(path3)


#第三个功能：进行预测建模，建立漏电电流和湿度、天气之间的关系，并根据明天晚上的天气来预测晚上8点到凌晨4点的漏电情况。
def loudian_decision_tree(path_train,path_test,loudian1,loudian2):
    data = pd.read_csv(path_train, encoding='utf-8')
    print('建立漏电电流和湿度、天气之间的关系时使用的数据量为：',data.shape[0])
    data.dropna(axis=0, how='any', inplace=True)
    
    X = data.iloc[:,[3,4,5,6,7,8]]    
    y = data.iloc[:, 9]
    #print(X.head(20))
    #print(y.head(20))
    
    X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
    
    dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
    
    
    #model=svm.SVC(C=5,kernel='poly',gamma=0.1,coef0=0,degree=1,probability = True)

    #alg = RandomForestClassifier(oob_score=False,random_state=8,n_estimators=50,max_depth=10,min_samples_split=2,min_samples_leaf=1,max_features=6)
  
    #model=svm.SVC(C=5,kernel='poly',gamma=0.1,coef0=0,degree=1,probability = True)
    
    #model.fit(X_train,y_train)
    #alg.fit(X_train,y_train)
    dtc.fit(X_train, y_train)    # 训练模型
    #model=model.fit(X_train,y_train)
    
    
    train_accuracy=dtc.score(X_train,y_train)
    validation_accuracy=dtc.score(X_test,y_test)
    print ('输出训练准确率：', train_accuracy)
    print ('输出验证准确率：', validation_accuracy)
    
    scores_cross_validation=model_selection.cross_val_score(dtc,X,y,cv=4)
    scores_mean=scores_cross_validation.mean()
    print('k折交叉验证的平均准确率：',scores_mean)
    
    
    if validation_accuracy<0.6:
        print('该线路漏电情况和天气等因素相关性为%.2f%%,还需要关注其他因素导致的电流漏电问题'%(validation_accuracy*100))
        #使用未知数据进行结果预测
        model=dtc.fit(X, y)
        data_test=pd.read_csv(path_test)
                
        #print('今晚和明晚的时间点对应的天气预报（包括：温度、湿度、天气）情况为：\n',data_test)
        
        #数据预处理
        data_test=pd.DataFrame(data_test)
        data_test.loc[data_test['weather'] == '雨', 'weather'] = 0
        data_test.loc[data_test['weather'] == '阵雨', 'weather'] = 0
    
        data_test.loc[data_test['weather'] == '阴', 'weather'] = 1
        data_test.loc[data_test['weather'] == '雾', 'weather'] = 2
        data_test.loc[data_test['weather'] == '多云', 'weather'] = 3
        data_test.loc[data_test['weather'] == '晴', 'weather'] = 4
    
        data_test.loc[data_test['weather'] == '阵雪', 'weather'] = 5
        data_test.loc[data_test['weather'] == '雪', 'weather'] = 5
        data_test.loc[data_test['weather'] == '雨夹雪', 'weather'] = 6
        data_test.loc[data_test['weather'] == '扬沙', 'weather'] = 7
    
    
        #风向编码，转化为算法可以处理的编码
        data_test.loc[data_test['wind'] == '东风', 'wind'] = 0
        data_test.loc[data_test['wind'] == '西风', 'wind'] = 1
        data_test.loc[data_test['wind'] == '南风', 'wind'] = 2
        data_test.loc[data_test['wind'] == '北风', 'wind'] = 3
        data_test.loc[data_test['wind'] == '东南风', 'wind'] = 4
        data_test.loc[data_test['wind'] == '东北风', 'wind'] = 5
        data_test.loc[data_test['wind'] == '西南风', 'wind'] = 6
        data_test.loc[data_test['wind'] == '西北风', 'wind'] = 7
        data_test.loc[data_test['wind'] == '无持续风向', 'wind'] = 8
        data_test.loc[data_test['wind'] == '风', 'wind'] = 9
        
        
        data_test['temperature']= data_test['temperature'].str.replace("℃","")
        data_test['humidity']=data_test['humidity'].str.replace("%","")
        data_test['winp']=data_test['winp'].str.replace("级","")
    
        data_test.dropna(inplace=True)
        data_test[['temperature','humidity','winp']]=data_test[['temperature','humidity','winp']].astype('int') 
        
        #print('测试数据为：\n',data_test)
        
        
        X_test= data_test.iloc[:,[1,2,3,4,5,6]]
        
        
        #测试数据的预测结果
        test_result=model.predict(X_test)
        print('预测结果：\n',test_result)
        predictions_proba=model.predict_proba(X_test)    
        predictions_proba1=predictions_proba.max(axis=1)
        #print(predictions_proba)
        print(predictions_proba1)
        
        
        
        time_weather=pd.read_csv(path_test)                
        print('今晚和明晚的时间点对应的天气预报（包括：温度、湿度、天气）情况为：\n',time_weather)
        real_result=test_result+1
        for i in range(len(test_result)):
            if test_result[i]==0:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第%d条数据漏电等级为：%d'%(i,real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[0],'--',loudian1[0])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==1:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[1],'--',loudian1[1])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==2:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[2],'--',loudian1[2])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==3:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[3],'--',loudian1[3])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==4:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[4],'--',loudian1[4])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
        
        
        
        real_result=data_test.iloc[:, 7]
        real_result=np.array(real_result)
        print('真实结果：\n',real_result)
        
        
        test_result_sum=0
        sum=len(test_result)
        for i in range(len(test_result)):
            if test_result[i]==real_result[i]:
                test_result_sum=test_result_sum+1
        test_accuracy=test_result_sum/sum
        print('测试数据的准确率为：',test_accuracy)
        
                
    elif 0.6<=validation_accuracy<0.8:
        print('该线路漏电电流值和天气、温湿度等因素相关性为%.2f%%'%(validation_accuracy*100))
        #使用未知数据进行结果预测
        model=dtc.fit(X, y)
        data_test=pd.read_csv(path_test)
        
        print('今晚和明晚的时间点对应的天气预报（包括：温度、湿度、天气）情况为：\n',data_test)
        
        #数据预处理
        data_test=pd.DataFrame(data_test)
        data_test.loc[data_test['weather'] == '雨', 'weather'] = 0
        data_test.loc[data_test['weather'] == '阵雨', 'weather'] = 0
    
        data_test.loc[data_test['weather'] == '阴', 'weather'] = 1
        data_test.loc[data_test['weather'] == '雾', 'weather'] = 2
        data_test.loc[data_test['weather'] == '多云', 'weather'] = 3
        data_test.loc[data_test['weather'] == '晴', 'weather'] = 4
    
        data_test.loc[data_test['weather'] == '阵雪', 'weather'] = 5
        data_test.loc[data_test['weather'] == '雪', 'weather'] = 5
        data_test.loc[data_test['weather'] == '雨夹雪', 'weather'] = 6
        data_test.loc[data_test['weather'] == '扬沙', 'weather'] = 7
    
    
        #风向编码，转化为算法可以处理的编码
        data_test.loc[data_test['wind'] == '东风', 'wind'] = 0
        data_test.loc[data_test['wind'] == '西风', 'wind'] = 1
        data_test.loc[data_test['wind'] == '南风', 'wind'] = 2
        data_test.loc[data_test['wind'] == '北风', 'wind'] = 3
        data_test.loc[data_test['wind'] == '东南风', 'wind'] = 4
        data_test.loc[data_test['wind'] == '东北风', 'wind'] = 5
        data_test.loc[data_test['wind'] == '西南风', 'wind'] = 6
        data_test.loc[data_test['wind'] == '西北风', 'wind'] = 7
        data_test.loc[data_test['wind'] == '无持续风向', 'wind'] = 8
        data_test.loc[data_test['wind'] == '风', 'wind'] = 9
        
        
        data_test['temperature']= data_test['temperature'].str.replace("℃","")
        data_test['humidity']=data_test['humidity'].str.replace("%","")
        data_test['winp']=data_test['winp'].str.replace("级","")
    
        data_test.dropna(inplace=True)
        data_test[['temperature','humidity','winp']]=data_test[['temperature','humidity','winp']].astype('int') 
        
        #print('测试数据为：\n',data_test)
        
    
        
        X_test= data_test.iloc[:,[1,2,3,4,5,6]]
        
       
        #测试数据的预测结果
        test_result=model.predict(X_test)
        print('预测结果：',test_result)
        predictions_proba=model.predict_proba(X_test)    
        predictions_proba1=predictions_proba.max(axis=1)
        print(predictions_proba)
        print(predictions_proba1)
        
        
        time_weather=pd.read_csv(path_test)                
        print('今晚和明晚的时间点对应的天气预报（包括：温度、湿度、天气）情况为：\n',time_weather)
        real_result=test_result+1
        for i in range(len(test_result)):
            if test_result[i]==0:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第%d条数据漏电等级为：%d'%(i,real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[0],'--',loudian1[0])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==1:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[1],'--',loudian1[1])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==2:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[2],'--',loudian1[2])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==3:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[3],'--',loudian1[3])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==4:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[4],'--',loudian1[4])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
        
        
        real_result=data_test.iloc[:, 7]
        real_result=np.array(real_result)
        print('真实结果：',real_result)
        
        test_result_sum=0
        sum=len(test_result)
        for i in range(len(test_result)):
            if test_result[i]==real_result[i]:
                test_result_sum=test_result_sum+1
        test_accuracy=test_result_sum/sum
        print('测试数据的准确率为：',test_accuracy)
       
    elif validation_accuracy>=0.8:
        print('该线路漏电电流值和天气、温湿度等因素关系密切,相关性为%.2f%%'%(scores_mean*100))
        
        
        #使用未知数据进行结果预测
        model=dtc.fit(X, y)
        data_test=pd.read_csv(path_test)
        
        print('今晚和明晚的时间点对应的天气预报（包括：温度、湿度、天气）情况为：\n',data_test)
        
        
        
        #数据预处理
        data_test=pd.DataFrame(data_test)
        data_test.loc[data_test['weather'] == '雨', 'weather'] = 0
        data_test.loc[data_test['weather'] == '阵雨', 'weather'] = 0
    
        data_test.loc[data_test['weather'] == '阴', 'weather'] = 1
        data_test.loc[data_test['weather'] == '雾', 'weather'] = 2
        data_test.loc[data_test['weather'] == '多云', 'weather'] = 3
        data_test.loc[data_test['weather'] == '晴', 'weather'] = 4
    
        data_test.loc[data_test['weather'] == '阵雪', 'weather'] = 5
        data_test.loc[data_test['weather'] == '雪', 'weather'] = 5
        data_test.loc[data_test['weather'] == '雨夹雪', 'weather'] = 6
        data_test.loc[data_test['weather'] == '扬沙', 'weather'] = 7
    
    
        #风向编码，转化为算法可以处理的编码
        data_test.loc[data_test['wind'] == '东风', 'wind'] = 0
        data_test.loc[data_test['wind'] == '西风', 'wind'] = 1
        data_test.loc[data_test['wind'] == '南风', 'wind'] = 2
        data_test.loc[data_test['wind'] == '北风', 'wind'] = 3
        data_test.loc[data_test['wind'] == '东南风', 'wind'] = 4
        data_test.loc[data_test['wind'] == '东北风', 'wind'] = 5
        data_test.loc[data_test['wind'] == '西南风', 'wind'] = 6
        data_test.loc[data_test['wind'] == '西北风', 'wind'] = 7
        data_test.loc[data_test['wind'] == '无持续风向', 'wind'] = 8
        data_test.loc[data_test['wind'] == '风', 'wind'] = 9
        
        
        data_test['temperature']= data_test['temperature'].str.replace("℃","")
        data_test['humidity']=data_test['humidity'].str.replace("%","")
        data_test['winp']=data_test['winp'].str.replace("级","")
    
        data_test.dropna(inplace=True)
        data_test[['temperature','humidity','winp']]=data_test[['temperature','humidity','winp']].astype('int') 
        
        #print('测试数据为：\n',data_test)
        
    
        X_test= data_test.iloc[:,[1,2,3,4,5,6]]
        
        #测试数据的预测结果
        test_result=model.predict(X_test)
        print('预测结果：',test_result)
        
        predictions_proba=model.predict_proba(X_test)    
        predictions_proba1=predictions_proba.max(axis=1)
        print(predictions_proba)
        print(predictions_proba1)
        
        
        time_weather=pd.read_csv(path_test)                
        print('今晚和明晚的时间点对应的天气预报（包括：温度、湿度、天气）情况为：\n',time_weather)
        real_result=test_result+1
        for i in range(len(test_result)):
            if test_result[i]==0:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第%d条数据漏电等级为：%d'%(i,real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[0],'--',loudian1[0])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==1:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[1],'--',loudian1[1])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==2:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[2],'--',loudian1[2])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==3:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[3],'--',loudian1[3])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
            if test_result[i]==4:
                real_result[i]=test_result[i]+1
                #print('根据明天每个时刻的天气预报和大数据建模，预测到该条线路的第'+str(i)+'个时刻漏电等级为：'+str(real_result[i]))
                real_time=time_weather.ix[i,:]
                print('时间和天气预报为：\n',real_time)
                print('根据该时刻的天气预报和大数据建模，预测到该条线路该时刻电流数据漏电等级可能为：%d'%real_result[i])
                print('漏电电流大致范围是：',loudian2[4],'--',loudian1[4])
                prob=predictions_proba[i].max()*100
                #print('发生的概率为:%.2f%%'%prob)
                print('\n')
        

        
        real_result=data_test.iloc[:, 7]
        real_result=np.array(real_result)
        print('真实结果：',real_result)
        
        test_result_sum=0
        sum=len(test_result)
        for i in range(len(test_result)):
            if test_result[i]==real_result[i]:
                test_result_sum=test_result_sum+1
        test_accuracy=test_result_sum/sum
        print('测试数据的准确率为：',test_accuracy)
                   
    
path1='E:\data_mining\loudian_problem\data_hefei\dataset8.csv'

path2='E:\data_mining\loudian_problem\data_hefei_yuchuli\dataset8_dropyuchuli1.csv'

path3='E:/data_mining/loudian_problem/data_hefei_yuchuli/dataset8_finalyuchuli2.csv'

path_train='E:/data_mining/loudian_problem/data_hefei_yuchuli/dataset8_finaltrain.csv'

path_test='E:\data_mining\loudian_problem\data_hefei_yuchuli\dataset8_test.csv'

loudian1,loudian2=fengxiantixing(path1)

loudian1,loudian2=loudiandengjihuafen(loudian1,loudian2)

data_preprocessing(path2,path3,loudian1,loudian2)

loudian_decision_tree(path_train,path_test,loudian1,loudian2)
    
    
