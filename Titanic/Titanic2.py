# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:30:52 2017

@author: jack.gang
"""

import pandas as pd
import csv
#from sklearn.cluster import KMeans
import statsmodels.formula.api as sm

path = 'train.csv'
pathUniv = open(path)
train = pd.read_csv(pathUniv, sep=',', engine='python')
pathUniv.close()

for index, row in train.iterrows():        
    if row['Age'] < 16 or row['Age'] > 75:
        train.set_value(index, 'Age', 1)
    else:
        train.set_value(index, 'Age', 0)

form = "Survived ~ C(Pclass)-1 + C(Sex) + C(Age) + SibSp"

model = sm.ols(form, data=train).fit()
print(model.summary())

# training
for index, row in train.iterrows():
    
    estimate = row['SibSp']*model.params['SibSp']
    
    if row['Age'] == 1:
        estimate += model.params['C(Age)[T.1.0]']
    
    if row['Sex'] == 'male':
        estimate += model.params['C(Sex)[T.male]']
        
    if row['Pclass'] == 1:
        estimate += model.params['C(Pclass)[1]']
    elif row['Pclass'] == 2:
        estimate += model.params['C(Pclass)[2]']
    else:
        estimate += model.params['C(Pclass)[3]']
#        
#    if row['Embarked'] == 'Q':
#        estimate += model.params['C(Embarked)[T.Q]']
#    elif row['Embarked'] == 'S':
#        estimate += model.params['C(Embarked)[T.S]']
    
    train.set_value(index, 'estimate', min(1, round(estimate)))

print("training:", (len(train) - sum(abs(train['Survived'] - train['estimate']))) / len(train))

# test
path = 'test.csv'
pathUniv = open(path)
test = pd.read_csv(pathUniv, sep=',', engine='python')
pathUniv.close()

for index, row in test.iterrows():        
    if row['Age'] < 16 or row['Age'] > 75:
        test.set_value(index, 'Age', 1)
    else:
        test.set_value(index, 'Age', 0)

for index, row in test.iterrows():
    
    estimate = row['SibSp']*model.params['SibSp']
    
    if row['Age'] == 1:
        estimate += model.params['C(Age)[T.1.0]']
    
    if row['Sex'] == 'male':
        estimate += model.params['C(Sex)[T.male]']
        
    if row['Pclass'] == 1:
        estimate += model.params['C(Pclass)[1]']
    elif row['Pclass'] == 2:
        estimate += model.params['C(Pclass)[2]']
    else:
        estimate += model.params['C(Pclass)[3]']
    
    test.set_value(index, 'Survived', min(1, round(estimate)))

test['Survived'] = test['Survived'].astype(int)
test[['PassengerId','Survived']].to_csv("result3.csv", header = ['PassengerId','Survived'], index = False, quoting = csv.QUOTE_NONE, quotechar = '')