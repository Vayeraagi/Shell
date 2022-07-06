# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:54:12 2022

@author: Mriank Ghosh
"""

import numpy
import matplotlib.pyplot as plt

mean = 0
std = 1 
num_samples = 10000000
samples = numpy.random.normal(mean, std, size=num_samples)
