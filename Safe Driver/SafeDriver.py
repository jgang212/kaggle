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
    
submission.to_csv("result2.csv", header = ['id','target'], index = False, quoting = csv.QUOTE_NONE, quotechar = '')
