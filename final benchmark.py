# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 10:20:44 2022

@author: Mriank Ghosh
"""

#this benchmarking consists of a list of 10 methods - 5 classical time series
# analysis methods + 5 ML adaptations for time series analysis. They will be 
# against a massive dataset of white noise. the expected results of the 
# benchmark are -- 
#1) No algorithm should have any significant advantage in forecasting
#2) lookout for their runtime and memory footprint.
#=============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import tracemalloc
from sklearn.metrics import mean_absolute_error as mae


#*****************************************************************************
#CLASSICAL TIME SERIES ANALYSIS METHODS--2
#*****************************************************************************

#ARIMA
from statsmodels.tsa.arima.model import ARIMA

# contrived dataset
data = np.load('C:/Users/Mriank Ghosh/Desktop/shell/whitenoise5M.npy')
l=5
train_data = data[:len(data)-l]
test_data = data[len(data)-l:]

#time and memory
start = time.time()
tracemalloc.start()
# fit model
model = ARIMA(train_data, order=(2, 1, 2))
model_fit = model.fit()
# make prediction
y = model_fit.predict(start=len(train_data), end=(len(data)-1), typ='levels')
pred = y[-l:]
realvalues = test_data[-l:]

error1 = mae(realvalues, pred)
print(error1)
end = time.time()
#t1=end-start
m1=tracemalloc.get_traced_memory()
#print("The time of execution of above program is :", t1)
print(m1)
tracemalloc.stop()
##############################################################################

#SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# contrived dataset
data = np.load('C:/Users/Mriank Ghosh/Desktop/shell/whitenoise3M.npy')
l=5
train_data = data[:len(data)-l]
test_data = data[len(data)-l:]

#time and memory
start = time.time()
tracemalloc.start()
# fit model
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 2))
model_fit = model.fit(disp=False)
# make prediction
y = model_fit.predict(start=len(train_data), end=(len(data)-1))

pred = y[-l:]
realvalues = test_data[-l:]

error2 = mae(realvalues, pred)
print(error2)
end = time.time()
t2=end-start
m2=tracemalloc.get_traced_memory()
print("The time of execution of above program is :", t2)
print(m2)
tracemalloc.stop()
##############################################################################

#VAR
from statsmodels.tsa.vector_ar.var_model import VAR
# contrived dataset
v1 = np.load('C:/Users/Mriank Ghosh/Desktop/shell/whitenoise3M.npy')
v2=np.load('C:/Users/Mriank Ghosh/Desktop/shell/whitenoise3M_2.npy')
data = list()
row = [v1, v2]
data.append(row)
#l=5
#train_data = data[:len(data)-l]
#test_data = data[len(data)-l:]
#time and memory
start = time.time()
tracemalloc.start()
# fit model
model = VAR(data)
model_fit = model.fit()
# make prediction
y = model_fit.forecast(model_fit.y, steps=1)

pred = y[-l:]
realvalues = test_data[-l:]

error2 = mae(realvalues, pred)
print(error2)
end = time.time()
t2=end-start
m2=tracemalloc.get_traced_memory()
print("The time of execution of above program is :", t2)
print(m2)
tracemalloc.stop()
#############################################################################

#VARMA
# VARMA example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)
#############################################################################

#HWES
# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(0, 100)]
# fit model
model = ExponentialSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
#############################################################################


#############################################################################
#graphing
#############################################################################

#runtime plot
import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'ARIMA(2,1,2)':t1, 'C++':15, 'Java':30,
        'Python':35}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()

#memory plot
