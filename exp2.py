# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:27:16 2022

@author: Mriank Ghosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import time
import tracemalloc

def naivePred(x,l):
    a = np.empty(l)
    for i in range(l):
        a[i] = np.mean(x)
        
    return a;

start = time.time()
tracemalloc.start()
#df = pd.read_csv('C:/Users/Mriank Ghosh/Desktop/shell/AMDfull.csv')
#print(df[df['Close'].isnull()])
#df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
#df = df.dropna(subset=['Close'])
#df.head()
x=np.load('C:/Users/Mriank Ghosh/Desktop/shell/whitenoise.npy')
#checking sttn
#adfuller(x)

l=5;
x_train = x[:-l]

y = naivePred(x_train,l)
#plt.plot(x, color = 'black')
#plt.plot(y, color = 'red')
#plt.xlabel("Days")
#plt.ylabel("Stock Price")
#plt.legend(["Real","Predicted"],loc="lower right")
#plt.show()
pred = y[-l:]
realvalues = x[-l:]
error = 100*(pred.T-realvalues)/realvalues

end = time.time()
 
print(abs(error))
print("The time of execution of above program is :", end-start)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
