# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:23:50 2022

@author: Mriank Ghosh
"""

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tracemalloc
import pandas as pd

# Plot 1: AR parameter = +0.9
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=80000000)
