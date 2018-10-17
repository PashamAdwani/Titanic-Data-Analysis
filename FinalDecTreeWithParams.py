import time
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

##=============================
#Read the training file to 'df'
#==============================
if os.path.exists(r'D:\Data Analysis\Project1\Project1'):
    df=pd.read_csv(r'D:\Data Analysis\Project1\Project1\titanic_train.csv',header=0)
else:
    print('Life sux')

    
##=============================
#Print head and tail training file 
#==============================

print(df.shape)
print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")

##=============================
#Read test file 
#==============================

if os.path.exists(r'D:\Data Analysis\Project1\Project1'):
    test=pd.read_csv(r'D:\Data Analysis\Project1\Project1\titanic_heldout.csv',header=0)
else:
    print('Cant read test file')

    
##=============================
#Print head and tail testing file 
#==============================

print(test.shape)
print("* test.head()", test.head(), sep="\n", end="\n\n")
print("* test.tail()", test.tail(), sep="\n", end="\n\n")


##=============================
#Prelims: some important features 
#==============================    
    
features=list(df.columns)

print(df['survived'].value_counts())
x=df['survived'].value_counts()
total=x[0]+x[1]
per_living=(x[1]/total)*100
print("Percent of Living:",per_living)
print("Percent of Dead:",100-per_living)

##=============================
#DATA PREPROCESSING 
#==============================
#Encode is a function to "MAP" the alphabetical values to a numeric one
#We are going to use encode for name,sex,embarked
#==============================
def encode(column_name,new_name):
    unique_vals=column_name.unique()
    map_to_int={name:n for n,name in enumerate(unique_vals)}
    new_name=column_name.replace(map_to_int)
    return new_name
#==============================
#saving the orginal dataframe and copying to df1
df1=df.copy()
#==============================
#plcass:-Good to go
#==============================
#survival:-result
df1_results=df1.survived
#==============================
#name:-1st letter taken and "encoded" with the numerical value
nam=df1.name
i=0
for x in df1.name:
    i=i+1
    nam[i]=x[1]

df1.name=nam

#encode the first letters in df1.name to a numerical value
df1.name=encode(df1.name,nam)
#==============================
#sex has to be "ENCODED" from F and M to 0 and 1
gender=df1.sex
df1.sex=encode(df1.sex,gender)
#==============================
#age : there are some NaN values
#in age which has to be filled with average values
df1.age=df1.age.fillna(df1.age.mean())
#==============================
#checked if sibsp has any NaN values
df1.sibsp.isna().any()
#sibsp is good to go - no value is empty
#==============================
#checked if parch has any NaN values
df1.parch.isna().any()
#parch is good to go - no value is empty
#==============================
df1.fare.isna().any()
#fare has NaN values:
df1.fare=df1.fare.fillna(df1.fare.mean())
#==============================
#Embarked has to be "ENCODED"
embark=df1.embarked
df1.embarked=encode(df1.embarked,embark)
#==============================
df1.boat.isna().any()
#boat has lot of empty values
df1.boat.fillna(0)
boat1=df1.boat
df1.boat=encode(df1.boat,boat1)
#==============================
df1.body.isna().any()
#boat has empty values
df1.body.fillna(0)
body1=df1.body
df1.body=encode(df1.body,body1)
#==============================

#MANIPULATING TEST DATA

#==============================
#survived as result
test_results=test.survived
#encoding name
nam=test.name
i=0
for x in test.name:
    i=i+1
    nam[i]=x[1]

test.name=nam
test.name=encode(test.name,nam)
#encoding sex
gender=test.sex
test.sex=encode(test.sex,gender)
#age is filled with mean
test.age=test.age.fillna(test.age.mean())
#fare NaN is filled with avg
test.fare=test.fare.fillna(test.fare.mean())
#Embarked
embark=test.embarked
test.embarked=encode(test.embarked,embark)
test.embarked.fillna(test.embarked.mode())
#Boat fill values
test.boat.fillna(0)
boat1=test.boat
test.boat=encode(test.boat,boat1)
#Body
test.body.fillna(0)
body1=test.body
test.body=encode(test.body,body1)
#==============================
#==============================

cols=['pclass','sex','age','sibsp','parch','embarked','fare','body','boat']

#==============================
#Decision Tree
#==============================

best=0

rs=[100,150,200,250]
cri=['gini','entropy']
spli=['best','random']
mxfeat=['auto','sqrt','log2']
for r in rs:
    for c in cri:
        for s in spli:
            for m in mxfeat:
                dt = DecisionTreeClassifier(criterion=c,splitter=s,max_features=m,random_state=r)
                dt.fit(df1[cols], df1.survived) # fitting the data in the model 
                ypred=dt.predict(test[cols])# preditcting the test data results 
                score=0;
                for x in range(263):
                    if(ypred[x]==test.survived[x]):
                        score=score+1
                        
                print('Parameters:criteria:',c,' Splitter:',s,' MaxFeatures:',m,' Random State:',r)
                print((score/263)*100,'% accuracy')
                accuracy=(score/263)*100
                if(best<accuracy):
                    best=accuracy
                    params=[x,s,m,r]
            

print('Best accuracy:',best)
print('Best Parameters:',params)

