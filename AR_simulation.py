# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:23:50 2022

@author: Mriank Ghosh
"""

from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
from memory_profiler import profile

# Plot 1: AR parameter = +0.9
@profile
def my_func():
    ar1 = np.array([1, -0.9])
    ma1 = np.array([1])
    AR_object1 = ArmaProcess(ar1, ma1)
    simulated_data_1 = AR_object1.generate_sample(nsample=85000000)

if __name__ == '__main__':
	my_func()
