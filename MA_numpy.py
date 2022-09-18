# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:04:23 2022

@author: Mriank Ghosh
"""

# MA example
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tracemalloc
from memory_profiler import profile
# instantiating the decorator
@profile
def my_func():
    #time and memory
    start = time.time()
    tracemalloc.start()
    # load dataset
    X = np.load('C:/Users/Mriank Ghosh/Desktop/shell/dataset/MA_1m.npy')
    # split dataset
    train_size = int(len(X) * 0.80)
    train, test = X[0:train_size], X[train_size:len(X)]
    #1 iteration****************************************
    # train moving average
    # fit model
    model = ARIMA(train, order=(0, 0, 1))
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

if __name__ == '__main__':
	my_func()