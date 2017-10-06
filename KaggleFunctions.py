# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:38:44 2017

@author: jack.gang
"""

import numpy as np

def fillMissingLinear(dFrame, columnName, isInt):
    # get average, std, and number of NaN values in dFrame
    avg = dFrame[columnName].mean()
    std = dFrame[columnName].std()
    cnt = dFrame[columnName].isnull().sum()
#    print(avg, std, cnt)
    
    # generate random numbers between (mean - std) & (mean + std)
    if isInt:
        rand = np.random.randint(avg - std, avg + std, size = cnt)
    else:
        rand = np.random.random(cnt) * 2 * std + avg - std
#    print(len(rand), rand.mean(), rand.std())                               
    
    # fill NaN values in columnName column with random values generated
    dFrame[columnName][np.isnan(dFrame[columnName])] = rand