# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:59:36 2022

@author: Mriank Ghosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import time
import tracemalloc


def lmsPred(x,l,u,N):
    xd= np.block([np.zeros((1,l)), x]).T
    y=np.zeros((len(xd),1))
    xn=np.zeros((N+1,1))
    xn = np.matrix(xn)
    wn=np.random.rand(N+1,1)/10
    M=len(xd)
    for n in range(0,M):
        xn = np.block([[xd[n]], [xn[0:N]]]);
        y[n]= np.matmul(wn.T, xn);
        if(n>M-l-1):
            e =0;
        else:
            e=int(x[n]-y[n]);
        wn = wn + 2*u*e*xn;
        
    return y,wn;

start = time.time()
tracemalloc.start()
#df = pd.read_csv('C:/Users/Mriank Ghosh/Desktop/shell/AMDfull.csv')
#print(df[df['Close'].isnull()])
#df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
#df = df.dropna(subset=['Close'])
#df.head()
x=np.load('C:/Users/Mriank Ghosh/Desktop/shell/whitenoise2.npy')
#checking sttn
#adfuller(x)

u = 2**(-30);
l=5;
N=30;
x_train = x[:-l]

y,wn = lmsPred(x_train,l,u,N)
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
