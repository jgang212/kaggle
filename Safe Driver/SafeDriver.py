# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:30:53 2017

@author: jack.gang
"""

# Imports
import csv
from os import chdir
import missingno as msno

# pandas
import pandas as pd
from pandas import DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression

#################################################################

# get train & test csv files as a DataFrame
chdir('E:\Jack Gang\Kaggle\Safe Driver')

driver_df = pd.read_csv("train.csv")
test_df   = pd.read_csv("test.csv")

#driver_df.info()
#print("----------------------------")
#test_df.info()
#
#print(driver_df.head())
#print(test_df.head())

#################################################################

# clean up missing data

# any() applied twice to check run the isnull check across all columns.
#driver_df.isnull().any().any()

driver_df_copy = driver_df.copy()
test_df_copy = test_df.copy()

driver_df_copy = driver_df_copy.replace(-1, np.NaN)
driver_df_copy.isnull().any().any()

test_df_copy = test_df_copy.replace(-1, np.NaN)
test_df_copy.isnull().any().any()

# Nullity or missing values by columns
#msno.bar(df=driver_df_copy, figsize=(20, 14), log = True)
#msno.bar(df=test_df_copy, figsize=(20, 14), log = True)

# fields with missing values (* means also missing in test)
# ps_ind_02_cat *
# ps_ind_04_cat *
# ps_ind_05_cat *
# ps_reg_03 *
# ps_car_01_cat *
# ps_car_02_cat *
# ps_car_03_cat *
# ps_car_05_cat *
# ps_car_07_cat *
# ps_car_09_cat *
# ps_car_11 *
# ps_car_12
# ps_car_14 *

# check -1 values
#driver_df_copy = driver_df_copy.replace(np.NaN, -1)
#test_df_copy = test_df_copy.replace(np.NaN, -1)

#sns.countplot(x='ps_ind_02_cat', data=driver_df_copy)
#sns.countplot(x='ps_ind_04_cat', data=driver_df_copy)
#sns.countplot(x='ps_ind_05_cat', data=driver_df_copy)
#sns.countplot(x='ps_car_01_cat', data=driver_df_copy)
#sns.countplot(x='ps_car_02_cat', data=driver_df_copy)
#sns.countplot(x='ps_car_03_cat', data=driver_df_copy)
#sns.countplot(x='ps_car_05_cat', data=driver_df_copy)
#sns.countplot(x='ps_car_07_cat', data=driver_df_copy)
#sns.countplot(x='ps_car_09_cat', data=driver_df_copy)

#driver_df_copy = driver_df_copy.replace(-1, np.NaN)
#test_df_copy = test_df_copy.replace(-1, np.NaN)

#driver_df_copy['ps_reg_03'].plot(kind='hist', figsize=(10,3),bins=100, xlim=(0,5))
fillMissingLinear(driver_df_copy, 'ps_reg_03', isInt = False)
fillMissingLinear(test_df_copy, 'ps_reg_03', isInt = False)
#driver_df_copy['ps_reg_03'].plot(kind='hist', figsize=(10,3),bins=100, xlim=(0,5))

#driver_df_copy['ps_car_11'].plot(kind='hist', figsize=(10,3),bins=100, xlim=(0,4))
fillMissingLinear(driver_df_copy, 'ps_car_11', isInt = True)
fillMissingLinear(test_df_copy, 'ps_car_11', isInt = True)
#driver_df_copy['ps_car_11'].plot(kind='hist', figsize=(10,3),bins=100, xlim=(0,4))

fillMissingLinear(driver_df_copy, 'ps_car_12', isInt = False)

#driver_df_copy['ps_car_14'].plot(kind='hist', figsize=(10,3),bins=100, xlim=(0,1))
fillMissingLinear(driver_df_copy, 'ps_car_14', isInt = False)
fillMissingLinear(test_df_copy, 'ps_car_14', isInt = False)
#driver_df_copy['ps_car_14'].plot(kind='hist', figsize=(10,3),bins=100, xlim=(0,1))

# fill ps_ind_02_cat missing values with the most occurred value, which is "1".
driver_df_copy["ps_ind_02_cat"] = driver_df_copy["ps_ind_02_cat"].fillna(1)
test_df_copy["ps_ind_02_cat"] = test_df_copy["ps_ind_02_cat"].fillna(1)

# fill ps_ind_04_cat missing values with the most occurred value, which is "0".
driver_df_copy["ps_ind_04_cat"] = driver_df_copy["ps_ind_04_cat"].fillna(0)
test_df_copy["ps_ind_04_cat"] = test_df_copy["ps_ind_04_cat"].fillna(0)

# fill ps_ind_05_cat missing values with the most occurred value, which is "0".
driver_df_copy["ps_ind_05_cat"] = driver_df_copy["ps_ind_05_cat"].fillna(0)
test_df_copy["ps_ind_05_cat"] = test_df_copy["ps_ind_05_cat"].fillna(0)

# fill ps_car_01_cat missing values with the most occurred value, which is "11".
driver_df_copy["ps_car_01_cat"] = driver_df_copy["ps_car_01_cat"].fillna(11)
test_df_copy["ps_car_01_cat"] = test_df_copy["ps_car_01_cat"].fillna(11)

# fill ps_car_02_cat missing values with the most occurred value, which is "1".
driver_df_copy["ps_car_02_cat"] = driver_df_copy["ps_car_02_cat"].fillna(1)
test_df_copy["ps_car_02_cat"] = test_df_copy["ps_car_02_cat"].fillna(1)

# fill ps_car_07_cat missing values with the most occurred value, which is "1".
driver_df_copy["ps_car_07_cat"] = driver_df_copy["ps_car_07_cat"].fillna(1)
test_df_copy["ps_car_07_cat"] = test_df_copy["ps_car_07_cat"].fillna(1)

# fill ps_car_09_cat missing values with the most occurred value, which is "2".
driver_df_copy["ps_car_09_cat"] = driver_df_copy["ps_car_09_cat"].fillna(2)
test_df_copy["ps_car_09_cat"] = test_df_copy["ps_car_09_cat"].fillna(2)

# drop ps_car_03_cat and ps_car_05_cat because it has too many missing values
driver_df_copy = driver_df_copy.drop(['ps_car_03_cat','ps_car_05_cat'], axis=1)
test_df_copy = test_df_copy.drop(['ps_car_03_cat','ps_car_05_cat'], axis=1)

driver_df = driver_df_copy.copy()
test_df = test_df_copy.copy()

#driver_df = driver_df.replace(-1, np.NaN)
#print(driver_df.isnull().any().any())
#
#test_df = test_df.replace(-1, np.NaN)
#print(test_df.isnull().any().any())

#################################################################

# look at binary variables

bin_col = [col for col in driver_df.columns if '_bin' in col]
zero_list = []
one_list = []
for col in bin_col:
    zero_list.append((driver_df[col]==0).sum())
    one_list.append((driver_df[col]==1).sum())
#    print(col, "% of 1s:", (driver_df[col]==1).sum()/((driver_df[col]==0).sum() + (driver_df[col]==1).sum()))
    
trace1 = go.Bar(
    x=bin_col,
    y=zero_list ,
    name='Zero count'
)
trace2 = go.Bar(
    x=bin_col,
    y=one_list,
    name='One count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of 1 and 0 in binary variables'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')

# remove variables that are < 1% binary
driver_df = driver_df.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'], axis=1)
test_df = test_df.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'], axis=1)

#################################################################

# look for relevant categorical variables

threshold = 0.005

for varName in [col for col in driver_df.columns if ('_bin' in col) or ('_cat' in col)]:
#    varName = 'ps_ind_18_bin'
    #fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
    
    #sns.countplot(x=varName, data=driver_df, ax=axis1)
    #sns.countplot(x='target', hue=varName, data=driver_df, order=[1,0], ax=axis2)
    
    # group by var, and get the mean for targeted people for each value in var
    var_perc = driver_df[[varName, "target"]].groupby([varName],as_index=False).mean()
    if (max(var_perc['target']) - min(var_perc['target'])) < threshold:
        print(varName, max(var_perc['target']) - min(var_perc['target']))
        driver_df = driver_df.drop([varName], axis=1)
        test_df = test_df.drop([varName], axis=1)        
    #sns.barplot(x=varName, y='target', data=var_perc,order=[0,1],ax=axis3)
    
#################################################################

# look for relevant continuous variables

threshold = 0.05

for varName in [col for col in driver_df.columns if ('_bin' not in col) and ('_cat' not in col)]:
    if varName in ['id','target']:
        continue

    # get fare for target and not target people 
    var_not_target = driver_df[varName][driver_df["target"] == 0]
    var_target     = driver_df[varName][driver_df["target"] == 1]
    
    # get average and std for fare of survived/not survived passengers
    avg_var = DataFrame([var_not_target.mean(), var_target.mean()])
    std_var = DataFrame([var_not_target.std(), var_target.std()])
    
    var_range = 2*driver_df[varName].std()
    diff_ratio = (max(avg_var[0]) - min(avg_var[0])) / var_range
    if diff_ratio < threshold:
        print(varName, diff_ratio)
        driver_df = driver_df.drop([varName], axis=1)
        test_df = test_df.drop([varName], axis=1)      
    
#    # plot
#    driver_df[varName].plot(kind='hist', figsize=(15,3),bins=100, xlim=(min(driver_df[varName]),max(driver_df[varName])))
#    avg_var.index.names = std_var.index.names = ["target"]
#    avg_var.plot(yerr=std_var,kind='bar',legend=False)

#################################################################

# define training and testing sets

X_train = driver_df.drop("target",axis=1)
Y_train = driver_df["target"]
X_test  = test_df.copy()

# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)[:,1]

logreg.score(X_train, Y_train)

# Support Vector Machines

#svc = SVC()
#
#svc.fit(X_train, Y_train)
#
#Y_pred = svc.predict(X_test)
#
#svc.score(X_train, Y_train)

## Random Forests
#
#random_forest = RandomForestClassifier(n_estimators=100)
#
#random_forest.fit(X_train, Y_train)
#
#Y_pred = random_forest.predict(X_test)
#
#random_forest.score(X_train, Y_train)

#knn = KNeighborsClassifier(n_neighbors = 3)
#
#knn.fit(X_train, Y_train)
#
#Y_pred = knn.predict(X_test)
#
#knn.score(X_train, Y_train)
#
## Gaussian Naive Bayes
#
#gaussian = GaussianNB()
#
#gaussian.fit(X_train, Y_train)
#
#Y_pred = gaussian.predict(X_test)
#
#gaussian.score(X_train, Y_train)

# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(driver_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df

submission = pd.DataFrame({"id": test_df["id"], "target": Y_pred})
    
submission.to_csv("result4.csv", header = ['id','target'], index = False, quoting = csv.QUOTE_NONE, quotechar = '')
