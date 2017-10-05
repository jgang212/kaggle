# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:30:52 2017

@author: jack.gang
"""

import pandas as pd
import csv
#from sklearn.cluster import KMeans
import statsmodels.formula.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'train.csv'
pathUniv = open(path)
train = pd.read_csv(pathUniv, sep=',', engine='python')
pathUniv.close()

#sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train);

for index, row in train.iterrows():        
    if row['Age'] < 16 or row['Age'] > 75:
        train.set_value(index, 'Age', 1)
    else:
        train.set_value(index, 'Age', 0)
    
    if row['Fare'] > 0:
        train.set_value(index, 'Fare', np.log(row['Fare']))

form = "Survived ~ C(Pclass)-1 + C(Sex) + C(Age) + SibSp + C(Pclass):C(Sex)"

model = sm.ols(form, data=train).fit()
print(model.summary())
plt.scatter(train['PassengerId'], model.resid)
train['estimate'] = model.fittedvalues
train['estimate'] = [min(1, round(x)) for x in train['estimate']]

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
    
    estimate = model.params['Intercept'] + row['SibSp']*model.params['SibSp']
    
    if row['Age'] == 1:
        estimate += model.params['C(Age)[T.1.0]']
    
    if row['Sex'] == 'male':
        estimate += model.params['C(Sex)[T.male]']
        
    if row['Pclass'] == 2:
        estimate += model.params['C(Pclass)[T.2]']
    elif row['Pclass'] == 3:
        estimate += model.params['C(Pclass)[T.3]']
    
    test.set_value(index, 'Survived', min(1, round(estimate)))

test['Survived'] = test['Survived'].astype(int)
test[['PassengerId','Survived']].to_csv("result5.csv", header = ['PassengerId','Survived'], index = False, quoting = csv.QUOTE_NONE, quotechar = '')