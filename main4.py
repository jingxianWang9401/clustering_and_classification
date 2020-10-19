# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:43:10 2020

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
import csv
import os


#第一个功能：根据历史数据实现漏电分析，以扇形图和表格行视展示，并给出风险提醒
def loudian_analyse(path,path_train):
    data=pd.read_csv(path)
    data=data[-data.leakage_current.isin([0])]#删除漏电数据为0的所有行
    X=data.leakage_current
    X=X.values.reshape(-1,1)
    data_number=X.shape[0]
    print('除去漏电电流为0的数据，数据集使用的漏电数据条数为：',data_number)   
    if data_number<1000:
        print('该条线路可用的有效数据量过少，低于1000条数据，参考性较小')

    kmeans=KMeans(n_clusters=5).fit(X)#构建并训练模型
    quantity = pd.Series(kmeans.labels_).value_counts()
    #print( "聚类后每个类别的样本数量\n", (quantity))
    
    sum=quantity[0]+quantity[1]+quantity[2]+quantity[3]+quantity[4]
    scale0=quantity[0]/sum
    scale1=quantity[1]/sum
    scale2=quantity[2]/sum
    scale3=quantity[3]/sum
    scale4=quantity[4]/sum
    #print('每个聚类类别所占数据集总量的比例为：\n',scale0,'\n',scale1,'\n',scale2,'\n', scale3,'\n',scale4)
    
    #获取聚类之后每个聚类中心的数据
    resSeries = pd.Series(kmeans.labels_)
    
    res0 = resSeries[resSeries.values == 0]
    #print("聚类后类别为0的数据\n",(data.iloc[res0.index]))
    data0=data.iloc[res0.index]
    data0_dianliu=data0.ix[:,4]
    max0_dianliu=max(data0_dianliu)
    min0_dianliu=min(data0_dianliu)
    #print('类别0的最小最大值为：',min0_dianliu,max0_dianliu)
    
    res1 = resSeries[resSeries.values == 1]
    #print("聚类后类别为1的数据\n",(data.iloc[res1.index]))
    data1=data.iloc[res1.index]
    data1_dianliu=data1.ix[:,4]
    max1_dianliu=max(data1_dianliu)
    min1_dianliu=min(data1_dianliu)
    #print('类别1的最大最小值为：',min1_dianliu,max1_dianliu)
    
    res2 = resSeries[resSeries.values == 2]
    #print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
    data2=data.iloc[res2.index]
    data2_dianliu=data2.ix[:,4]
    max2_dianliu=max(data2_dianliu)
    min2_dianliu=min(data2_dianliu)
    #print('类别2的最大最小值为：',min2_dianliu,max2_dianliu)
    
    res3 = resSeries[resSeries.values == 3]
    #print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
    data3=data.iloc[res3.index]
    data3_dianliu=data3.ix[:,4]
    max3_dianliu=max(data3_dianliu)
    min3_dianliu=min(data3_dianliu)
    #print('类别3的最大最小值为：',min3_dianliu,max3_dianliu)
    
    res4 = resSeries[resSeries.values == 4]
    #print("聚类后类别为2的数据\n",(data.iloc[res2.index]))
    data4=data.iloc[res4.index]
    data4_dianliu=data4.ix[:,4]
    max4_dianliu=max(data4_dianliu)
    min4_dianliu=min(data4_dianliu)
    #print('类别4的最大最小值为：',min4_dianliu,max4_dianliu)
    
    
    loudian1=[max0_dianliu,max1_dianliu,max2_dianliu,max3_dianliu,max4_dianliu]
    loudian2=[min0_dianliu,min1_dianliu,min2_dianliu,min3_dianliu,min4_dianliu]
       

    # 生成数据
    labels=['漏电电流：%d--%d'%(min0_dianliu,max0_dianliu),'漏电电流：%d--%d'%(min1_dianliu,max1_dianliu),'漏电电流：%d--%d'%(min2_dianliu,max2_dianliu),'漏电电流：%d--%d'%(min3_dianliu,max3_dianliu),'漏电电流：%d--%d'%(min4_dianliu,max4_dianliu)]
    share = [scale0, scale1, scale2,scale3,scale4]
    
    figure_data=[[min0_dianliu,max0_dianliu,scale0],[min1_dianliu,max1_dianliu,scale1],[min2_dianliu,max2_dianliu,scale2],[min3_dianliu,max3_dianliu,scale3],[min4_dianliu,max4_dianliu,scale4]]
    
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
        
        
    loudian1.sort()
    loudian2.sort()
    print('根据本条回路的历史漏电数据进行聚类建模分析，该条回路可分为五个不同等级：')
    print('等级1：',loudian2[0],'--',loudian1[0])
    print('等级2：',loudian2[1],'--',loudian1[1])
    print('等级3：',loudian2[2],'--',loudian1[2])
    print('等级4：',loudian2[3],'--',loudian1[3])
    print('等级5：',loudian2[4],'--',loudian1[4])
    
    
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

    data['leakage_current']= data['leakage_current'].apply(lambda x: current_coding(x))
    
    
    df=pd.DataFrame(data)
    
    df.to_csv(path_train)
        
    return data_number,max_loudian,min_loudian,loudian1,loudian2,figure_data



#第二个功能：进行预测建模，建立漏电电流和湿度、天气之间的关系，并根据明天晚上的天气来预测晚上8点到凌晨4点的漏电情况。
def loudian_decision_tree(data_train,path_test,loudian1,loudian2):
    data = pd.read_csv(data_train)
    data.dropna(axis=0, how='any', inplace=True)
    
    X = data.iloc[:,[2,3,4]]    
    y = data.iloc[:, 5]
    #print(X.head(20))
    #print(y.head(20))
    
    X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=0)
    
    dtc = DTC(criterion='entropy')    # 初始化决策树对象，基于信息熵
    dtc.fit(X_train, y_train)    # 训练模型
      
    train_accuracy=dtc.score(X_train,y_train)
    validation_accuracy=dtc.score(X_test,y_test)
    print ('输出训练准确率：', train_accuracy)
    print ('输出验证准确率：', validation_accuracy)
    
    scores_cross_validation=model_selection.cross_val_score(dtc,X,y,cv=4)
    scores_mean=scores_cross_validation.mean()
    print('k折交叉验证的平均准确率：',scores_mean)
    
    model=dtc.fit(X, y)
    
    data_test=pd.read_csv(path_test)
    
    X_test= data_test.iloc[:,[1,2,3]]
    
    test_result=model.predict(X_test)
        
    print('预测结果：\n',test_result)
    
    grade=test_result
    for i in range(len(test_result)):
        grade[i]=test_result[i]+1
    
    print(grade)
    min=[1,2,3,4,5,6,7]
    max=[1,2,3,4,5,6,7]
    for i in range(len(grade)):
        #min.append(loudian2[i-1])
        #max.append(loudian2[i-1])
        
        if grade[i]==1:
            leakage_min=loudian2[0]
            leakage_max=loudian1[0]
            min[i]=leakage_min
            max[i]=leakage_max
            #print(min[i])
            #min.append(loudian2[i-1])
            #max.append(loudian2[0])
            #print(min,max)
            
        if grade[i]==2:
            leakage_min=loudian2[1]
            leakage_max=loudian1[1]
            min[i]=leakage_min
            max[i]=leakage_max
            #print(min[i])
            #min.append(loudian2[i-1])
            #max.append(loudian2[i-1])
            #print(min,max)
            #print(min,max)
        
        if grade[i]==3:
            leakage_min=loudian2[2]
            leakage_max=loudian1[2]
            min[i]=leakage_min
            max[i]=leakage_max
            #min.append(leakage_min)
            #max.append(leakage_max)
            #print(min,max)
            
        
        if grade[i]==4:
            leakage_min=loudian2[3]
            leakage_max=loudian1[3]
            min[i]=leakage_min
            max[i]=leakage_max
            #min.append(leakage_min)
            #max.append(leakage_max)
            #print(min,max)
        
        if grade[i]==5:
            leakage_min=loudian2[4]
            leakage_max=loudian1[4]
            min[i]=leakage_min
            max[i]=leakage_max
            #min.append(leakage_min)
            #max.append(leakage_max)
            #print(min,max)
           
    print(grade,min,max)
    return validation_accuracy,grade,min,max



def main():
    #分析
    result_path1="E:/data_mining/loudian_problem/data_out/analyse_result.csv"
    
    if os.path.exists(result_path1):  # 如果文件存在   
        os.remove(result_path1)    
    
    
    with open(result_path1, "a", newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["rtu", "loop","data_number","min_loudian","max_loudian", "min0","max0","scale0","min1","max1","scale1","min2","max2","scale2","min3","max3","scale3","min4","max4","scale4","loudian2[0]","loudian1[0]","loudian2[1]","loudian1[1]","loudian2[2]","loudian1[2]","loudian2[3]","loudian1[3]","loudian2[4]","loudian1[4]"])
    
    dataset_path='E:/data_mining/loudian_problem/data_in/'
        
    datasetlist=[]
    datasetlist=os.listdir(dataset_path)
    #print(datasetlist)
    #print(len(datasetlist))
    #分析+预测
    
    #预测
    result_path2="E:/data_mining/loudian_problem/data_out/predict_result.csv"
    
    if os.path.exists(result_path2):  # 如果文件存在   
        os.remove(result_path2)    
        
        
    with open(result_path2, "a", newline='') as f:
        
        writer = csv.writer(f)
        writer.writerow(["rtu","loop","correlation", "today_18_grade","today_18_min","today_18_max","today_20_grade", "today_20_min","today_20_max","today_22_grade","today_22_min","today_22_max","today_24_grade","today_24_min","today_24_max","tomorrow_2_grade","tomorrow_2_min","tomorrow_2_max","tomorrow_4_grade","tomorrow_4_min","tomorrow_4_max","tomorrow_6_grade","tomorrow_6_min","tomorrow_6_max"])
    
        
    for i in range(len(datasetlist)):
        filename=datasetlist[i]
        rtu_id=filename.split('_')[0]
        loop=filename.split('_')[1]
        loop_id=loop.split('.')[0]
        
        path='E:/data_mining/loudian_problem/data_in/'+filename#数据集读取，csv格式
        #print(data)
        path_train='E:/data_mining/loudian_problem/data_train/'+filename
        
        data_number,max_loudian,min_loudian,loudian1,loudian2,figure_data=loudian_analyse(path,path_train)

        
        with open("E:/data_mining/loudian_problem/data_out/analyse_result.csv", "a", newline='') as f:
            writer = csv.writer(f)
            #writer.writerow(["URL", "rrr","predict", "score"])
            row = [[rtu_id,loop_id, data_number, min_loudian,max_loudian,figure_data[0][0],figure_data[0][1],figure_data[0][2],figure_data[1][0],figure_data[1][1],figure_data[1][2],figure_data[2][0],figure_data[2][1],figure_data[2][2],figure_data[3][0],figure_data[3][1],figure_data[3][2],figure_data[4][0],figure_data[4][1],figure_data[4][2],loudian2[0],loudian1[0],loudian2[1],loudian1[1],loudian2[2],loudian1[2],loudian2[3],loudian1[3],loudian2[4],loudian1[4]]]
            for r in row:
                writer.writerow(r)
        

        
                
        dataset_train_path='E:/data_mining/loudian_problem/data_train/'
        dataset_test_path='E:/data_mining/loudian_problem/weather/'
        
        
        datasetlist1=[]
        datasetlist1=os.listdir(dataset_train_path)
        #print(datasetlist)
        #print(len(datasetlist))
        
        datasetlist2=[]
        datasetlist2=os.listdir(dataset_test_path)
        
        
        filename1=datasetlist1[i]
        rtu_id=filename1.split('_')[0]
        loop=filename1.split('_')[1]
        loop_id=loop.split('.')[0]
        
        filename2=datasetlist2[i]
        
        train_path='E:/data_mining/loudian_problem/data_train/'+filename1#数据集读取，csv格式
        #print(data)
        test_path='E:/data_mining/loudian_problem/weather/'+filename2
        
        validation_accuracy,grade,min,max=loudian_decision_tree(train_path,test_path,loudian1,loudian2) 
        
        
        with open("E:/data_mining/loudian_problem/data_out/predict_result.csv", "a", newline='') as f:
            writer = csv.writer(f)
            #writer.writerow(["URL", "rrr","predict", "score"])
            row = [[rtu_id,loop_id,validation_accuracy,grade[0], min[0], max[0],grade[1], min[1], max[1],grade[2], min[2], max[2],grade[3], min[3], max[3],grade[4], min[4], max[4],grade[5], min[5], max[5],grade[6], min[6], max[6]]]
            for r in row:
                writer.writerow(r)
                                        
main()
