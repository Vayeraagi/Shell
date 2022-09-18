# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:17:28 2022

@author: Mriank Ghosh
"""

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tracemalloc
import pandas as pd
import datatable as dt

#loading dataset
X  = dt.Frame(np.load('C:/Users/Mriank Ghosh/Desktop/shell/dataset/numpy_10m.npy'))
# split dataset
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:len(X)]
#1 iteration****************************************
#time and memory
start = time.time()
tracemalloc.start()
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
end = time.time()
t1=end-start
m1=tracemalloc.get_traced_memory()
d1=m1[1]-m1[0]
print("The time of execution of above program is :", t1/60)
print("(current memory usage) :",m1[0]/1000000)
print("(peak memory usage) :",m1[1]/1000000)
tracemalloc.stop()
#test accuracy
error1 = mean_squared_error(test, predictions)
print(error1)