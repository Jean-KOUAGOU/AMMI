# -*- coding: utf-8 -*-

import numpy as np
#import random as rd
#import seaborn as sn
import pandas as pd
#import matplotlib.pyplot as plt
import Jean_Percept_LogReg
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

Data=pd.read_csv('Electricity-Problem/electricity-normalized.csv')
Data.head()
le=preprocessing.LabelEncoder()
Data['class']=le.fit_transform(Data['class'])
X=Data.iloc[:,:-1]
Target=Data.iloc[:,-1]
X_train, X_test, target_train, target_test = train_test_split(X, Target, test_size=0.4, random_state=101)

def KFold_(data, k=10):
    L=[i for i in range(len(data))]
    np.random.shuffle(L)
    Data_list=[]
    m=len(data)
    for j in range(1,k+1):
        Data_list.append(data.iloc[L[int((j-1)*(m/k)):int(j*(m/k))]])
    Accuracy=[]
    for i in range(k):
        Validation=Data_list[i]
        Data=pd.DataFrame()
        for j in range(k):
            if j!=i:
                Data=pd.concat([Data, Data_list[j]])         
        theta=np.zeros(1+Data.iloc[:,:-1].shape[1])
        Log_Reg=Jean_Percept_LogReg.Jean_logisticRegression(Data.iloc[:,:-1], Data.iloc[:,-1], theta, verbose=True)
        Intercept_column=np.ones_like(Data.iloc[:,-1]).reshape(-1,1)
        Log_Reg.add_intercept(Intercept_column)
        Accuracy.append((Log_Reg.predict(Validation.iloc[:,:-1])==Validation.iloc[:,-1]).sum()/len(Validation.iloc[:,-1]))
        #print(Accuracy)
    accuracy=sum(Accuracy)/len(Accuracy)
    return accuracy

def BackwardWrapperStart(F, feature):
    if feature in F:
        F_feature=F.drop(feature)
        return F_feature

def BackwardWrapper(data, num_feature):
    F=data.columns.drop(data.columns[-1])
    while(len(F)>num_feature):
        accuracy_prev=0.0
        for feature in data.columns.drop(data.columns[-1]):
            if feature in F:
                F_feature=BackwardWrapperStart(F, feature)
                accuracy=KFold_(data[F_feature.insert(len(F_feature),data.columns[-1])])
                if accuracy>accuracy_prev:
                    F=F_feature
                    accuracy_prev=accuracy
                    print(accuracy_prev)
                else:
                    pass
            else:
                continue
    return F

DATA=pd.concat([X_train, target_train], axis=1)
print(BackwardWrapper(DATA, 4))