# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:14:24 2022

@author: Mriank Ghosh
"""

#Autoregressive (AR)
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tracemalloc
import h5py
from memory_profiler import profile
# instantiating the decorator
@profile
def my_func():
    #time and memory
    start = time.time()
    tracemalloc.start()
    #reading hdf5 file
    hf = h5py.File('1.hdf5', 'r')
    hf.keys()
    train,test = np.array(hf.get('train')),np.array(hf.get('test'))

    #1 iteration****************************************
    # train autoregression
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