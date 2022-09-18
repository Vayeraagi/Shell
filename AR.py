# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 21:24:03 2022

@author: Mriank Ghosh
"""
#Autoregressive (AR)
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tracemalloc
#time and memory
start = time.time()
tracemalloc.start()
# load dataset
X = np.load('C:/Users/Mriank Ghosh/Desktop/shell/dataset/AR_75m.npy')
# split dataset
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:len(X)]
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
#2 iteration****************************************
#time and memory
start = time.time()
tracemalloc.start()
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
end = time.time()
t2=end-start
m2=tracemalloc.get_traced_memory()
d2=m2[1]-m2[0]
print("The time of execution of above program is :", t2)
print("(current memory usage,peak memory usage) :",d2)
tracemalloc.stop()
#3 iteration****************************************
#time and memory
start = time.time()
tracemalloc.start()
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
end = time.time()
t3=end-start
m3=tracemalloc.get_traced_memory()
d3=m3[1]-m1[0]
print("The time of execution of above program is :", t3)
print("(current memory usage,peak memory usage) :",d3)
tracemalloc.stop()

#Mean
T=(t1+t2+t3)/3
T
M=(d1+d2+d3)/3
M
