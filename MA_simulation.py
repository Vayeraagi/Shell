# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:10:51 2022

@author: Mriank Ghosh
"""

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tracemalloc

# Plot 1: MA parameter: -0.9
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=8000000)
