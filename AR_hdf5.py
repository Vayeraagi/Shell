# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:57:19 2022

@author: Mriank Ghosh
"""

#Autoregressive (AR)
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import time
import tracemalloc
import pandas as pd
import h5py
#time and memory
start = time.time()
tracemalloc.start()
#reading hdf5 file
hf = h5py.File('80.hdf5', 'r')
hf.keys()
train = np.array(hf.get('train'))
test = np.array(hf.get('test'))

#1 iteration****************************************
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
#test accuracy
error1 = mean_squared_error(test, predictions)
print(error1)
#metrics
end = time.time()
t1=end-start
m1=tracemalloc.get_traced_memory()
d1=m1[1]-m1[0]
print("The time of execution of above program is :", t1/60)
print("(current memory usage) :",m1[0]/1000000)
print("(peak memory usage) :",m1[1]/1000000)
print("(memory usage) :",d1/1000000)
tracemalloc.stop()

