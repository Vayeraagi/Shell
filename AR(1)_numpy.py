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

#gnerating artificial data
mean = 0
std = 1 
num_samples = 10000000
samples = np.random.normal(mean, std, size=num_samples)
# load dataset
X = samples
