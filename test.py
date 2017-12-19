#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:49:15 2017

@author: janny
"""
import pandas as pd 
import numpy as np
from pandas import Series,DataFrame

data_train = pd.read_csv("/mnt/share/Titanic/data/train.csv")
data_train.info()

################################################  know the features
import matplotlib.pyplot as plt
plt.subplot2grid((2,3),(0,0))             
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Survived (1 is alive)") 
plt.ylabel(u"number")
###测试添加的代码git


plt.subplot2grid((2,3),(0,1))      #船舱的人数分布     
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"deng ji") 
plt.ylabel(u"number") 

plt.subplot2grid((2,3),(0,2))    #年龄与是否生存       
data_train.Age[data_train.Survived==0].plot(kind="kde",color='green')
data_train.Age[data_train.Survived==1].plot(kind="kde",color='r')
plt.title(u"Age distrubution of Survived (1 is alive)") 
plt.xlabel(u"Age") 
plt.legend((u'0', u'1'),loc='best')


plt.subplot2grid((2,3),(1,0),colspan=2)  #年齡與船舱等级
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")
plt.ylabel(u"density") 
plt.title(u"Age distrubution of Pclass ")
plt.legend((u'1', u'2',u'3'),loc='best')

#船舱等级与是否获救
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
pclassSurvived=pd.DataFrame({'Survived':Survived_1,'Died':Survived_0})
pclassSurvived.plot(kind="bar")
plt.xlabel(u"Pclass_level")

#性别与是否获救
Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
sexSurvived=pd.DataFrame({'Survived':Survived_1,'Died':Survived_0})
sexSurvived.plot(kind="bar")
plt.xlabel(u"sex of Survived")


#女性在不同等级舱的生存情况
Survived_0 = data_train.Pclass[data_train.Survived == 0][data_train.Sex=='female'].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1][data_train.Sex=='female'].value_counts()
femalePclass=pd.DataFrame({'Survived':Survived_1,'Died':Survived_0})
femalePclass.plot(kind="bar")
plt.xlabel(u"female in each Pclass")

#男性在不同等级舱的生存情况
Survived_0 = data_train.Pclass[data_train.Survived == 0][data_train.Sex=='male'].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1][data_train.Sex=='male'].value_counts()
malePclass=pd.DataFrame({'Survived':Survived_1,'Died':Survived_0})
malePclass.plot(kind="bar")
plt.xlabel(u"male in each Pclass")


#兄妹在船人数与生存情况
Survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts()
Survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
test=pd.DataFrame({'Survived':Survived_1,'Died':Survived_0})
test.plot(kind="bar")

#父母在船人数与生存情况
Survived_0 = data_train.Parch[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Parch[data_train.Survived == 1].value_counts()
test=pd.DataFrame({'Survived':Survived_1,'Died':Survived_0})
test.plot(kind="bar")

#是否有船舱信息与生存情况
cabinNotnull=data_train.Survived[data_train.Cabin.notnull()].value_counts()
cabinNull=data_train.Survived[data_train.Cabin.isnull()].value_counts()
cabinSupvived=pd.DataFrame({'cabinNotnull':cabinNotnull,'cabinNull':cabinNull})
cabinSupvived.plot(kind='bar')

#############################################################  processing
missNum=len(data_train.loc[data_train.Age.isnull(),'Age'])
sub=data_train.Age.max()-data_train.Age.min()
sum=setp=sub/missNum
Age_new=np.float64(np.arange(missNum))
for i in range(missNum):
    Age_new[i]=sum
    sum+=setp
data_train.loc[data_train.Age.isnull(),'Age']=Age_new
         
data_train.loc[data_train.Cabin.notnull(),'Cabin']='1'
data_train.loc[data_train.Cabin.isnull(),'Cabin']='0'
dummies_Cabin=pd.get_dummies(data_train.Cabin,prefix='Cabin')
dummies_Sex=pd.get_dummies(data_train.Sex,prefix='Sex')
dummies_Pclass=pd.get_dummies(data_train.Pclass,prefix='Pclass')
dummies_Embarked=pd.get_dummies(data_train.Embarked,prefix='Embarked')
test_Scalled=pd.concat([data_train,dummies_Cabin,dummies_Sex,dummies_Pclass,dummies_Embarked],axis=1)
test_Scalled.drop(['Cabin','Sex','Pclass','Embarked','Name','Ticket'],axis=1,inplace=True)

import sklearn.preprocessing as prepro 
scaler=prepro.StandardScaler()
data=scaler.fit(test_Scalled.Age)
test_Scalled['Age_scaled']=scaler.fit_transform(test_Scalled.Age,data)

data=scaler.fit(test_Scalled.Fare)
test_Scalled['Fare_scaled']=scaler.fit_transform(test_Scalled.Fare,data)

all_Data=test_Scalled
all_Data.drop(['PassengerId','Age','Fare'],axis=1,inplace=True)
all_Data=all_Data.as_matrix()

from sklearn import cross_validation
xTrain,xTest,yTrain,yTest=cross_validation.train_test_split(all_Data[:,1:],all_Data[:,0],test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(max_features=8,n_estimators=200,n_jobs=4)
clf.fit(xTrain,yTrain)
pred=clf.predict(xTest)

from sklearn import metrics
print(metrics.roc_auc_score(pred,yTest))

print(cross_validation.cross_val_score(clf,all_Data[:,1:],all_Data[:,0],n_jobs=4,cv=10))


###########  processes test data
#==============================================================================
# data_test = pd.read_csv("/mnt/share/Titanic/data/test.csv")
# missNum=len(data_test.loc[data_test.Age.isnull(),'Age'])
# sub=data_test.Age.max()-data_test.Age.min()
# sum=setp=sub/missNum
# Age_new=np.float64(np.arange(missNum))
# for i in range(missNum):
#     Age_new[i]=sum
#     sum+=setp
# data_test.loc[data_test.Age.isnull(),'Age']=Age_new
# data_test.loc[data_test.Cabin.notnull(),'Cabin']='1'
# data_test.loc[data_test.Cabin.isnull(),'Cabin']='0'
# 
# dummies_Cabin=pd.get_dummies(data_test.Cabin,prefix='Cabin')
# dummies_Sex=pd.get_dummies(data_test.Sex,prefix='Sex')
# dummies_Pclass=pd.get_dummies(data_test.Pclass,prefix='Pclass')
# dummies_Embarked=pd.get_dummies(data_test.Embarked,prefix='Embarked')
# test_Scalled=pd.concat([data_test,dummies_Cabin,dummies_Sex,dummies_Pclass,dummies_Embarked],axis=1)
# test_Scalled.drop(['Cabin','Sex','Pclass','Embarked','Name','Ticket'],axis=1,inplace=True)
# test_Scalled.loc[test_Scalled.Fare.isnull(),'Fare']=test_Scalled.Fare.mean()              
# 
# 
# data=scaler.fit(test_Scalled.Age)
# test_Scalled['Age_scaled']=scaler.fit_transform(test_Scalled.Age,data)
# 
# test_Scalled.loc[test_Scalled.Fare.isnull(),'Fare']=test_Scalled.Fare.mean()              
# data=scaler.fit(test_Scalled.Fare)
# test_Scalled['Fare_scaled']=scaler.fit_transform(test_Scalled.Fare,data)
# 
# all_TestData=test_Scalled
# all_TestData.drop(['PassengerId','Age','Fare'],axis=1,inplace=True)
# all_TestData=all_TestData.as_matrix()
# clf.fit(all_Data[:,1:],all_Data[:,0])
# pred2=clf.predict(all_TestData)
#               
# result=pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':pred2.astype(np.int32)})              
# result.to_csv('/mnt/share/Titanic/result.csv',index=False)
#==============================================================================
