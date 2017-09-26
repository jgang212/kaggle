# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:30:52 2017

@author: jack.gang
"""

import pandas as pd
from patsy import dmatrices
import numpy as np
import csv

path = 'train.csv'
pathUniv = open(path)
train = pd.read_csv(pathUniv, sep=',', engine='python')
pathUniv.close()

# clean up data
meanAge = np.nanmean(train['Age'])

for index, row in train.iterrows():
    if row['Age'] != row['Age']:
        train.set_value(index, 'Age', meanAge)
#    if row['Parch'] > 0:
#        train.set_value(index, 'Age', 0)
#    if row['SibSp'] > 0:
#        train.set_value(index, 'SibSp', 1)
#    if row['Parch'] > 0:
#        train.set_value(index, 'Parch', 1)

# model
outcome, predictors = dmatrices("Survived ~ C(Pclass)-1 + C(Sex) + Age + SibSp + Parch + C(Embarked) + Fare", train)

betas = np.linalg.lstsq(predictors, outcome)[0].ravel()
betaDict = {}
for name, beta in zip(predictors.design_info.column_names, betas):
    betaDict[name] = beta

# training
for index, row in train.iterrows():
    
    estimate = row['Age']*betaDict['Age'] + row['SibSp']*betaDict['SibSp'] + row['Parch']*betaDict['Parch'] + row['Fare']*betaDict['Fare']
    
    if row['Sex'] == 'male':
        estimate += betaDict['C(Sex)[T.male]']
        
    if row['Pclass'] == 1:
        estimate += betaDict['C(Pclass)[1]']
    elif row['Pclass'] == 2:
        estimate += betaDict['C(Pclass)[2]']
    else:
        estimate += betaDict['C(Pclass)[3]']
        
    if row['Embarked'] == 'Q':
        estimate += betaDict['C(Embarked)[T.Q]']
    elif row['Embarked'] == 'S':
        estimate += betaDict['C(Embarked)[T.S]']
    
    train.set_value(index, 'estimate', min(1, round(estimate)))

print("training:", (len(train) - sum(abs(train['Survived'] - train['estimate']))) / len(train))

# test
path = 'test.csv'
pathUniv = open(path)
test = pd.read_csv(pathUniv, sep=',', engine='python')
pathUniv.close()

for index, row in test.iterrows():
    
    estimate = row['Age']*betaDict['Age'] + row['SibSp']*betaDict['SibSp'] + row['Parch']*betaDict['Parch'] + row['Fare']*betaDict['Fare']
    
    if row['Sex'] == 'male':
        estimate += betaDict['C(Sex)[T.male]']
        
    if row['Pclass'] == 1:
        estimate += betaDict['C(Pclass)[1]']
    elif row['Pclass'] == 2:
        estimate += betaDict['C(Pclass)[2]']
    else:
        estimate += betaDict['C(Pclass)[3]']
        
    if row['Embarked'] == 'Q':
        estimate += betaDict['C(Embarked)[T.Q]']
    elif row['Embarked'] == 'S':
        estimate += betaDict['C(Embarked)[T.S]']
    
    test.set_value(index, 'Survived', min(1, round(estimate)))

test['Survived'] = test['Survived'].astype(int)
test[['PassengerId','Survived']].to_csv("result2.csv", header = ['PassengerId','Survived'], index = False, quoting = csv.QUOTE_NONE, quotechar = '')
