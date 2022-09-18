# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 11:02:58 2022

@author: Mriank Ghosh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import time
import tracemalloc

#generating artificial data
mean = 0
std = 1 
num_samples = 90000000
samples = np.random.normal(mean, std, size=num_samples)
df = pd.DataFrame(samples, columns = ['data'])
#time and memory
start = time.time()
tracemalloc.start()
# data preprocess
df["data_shifted"]  = df["data"].shift()
df.head()
df.dropna(inplace=True)
df.head()
# load dataset
y = df.data.values
x = df.data_shifted.values
# split dataset
train_size = int(len(x) * 0.80)
x_train, x_test = x[0:train_size], x[train_size:len(x)]
y_train, y_test = y[0:train_size], y[train_size:len(x)]
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
#fit
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
#record 
end = time.time()
t=end-start
m=tracemalloc.get_traced_memory()
d=m[1]-m[0]
print("The time of execution of above program is :", t)
print("(current memory usage,peak memory usage) :",d)
tracemalloc.stop()
