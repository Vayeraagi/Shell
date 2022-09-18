# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:04:20 2022

@author: Mriank Ghosh
"""

import numpy as np
import h5py

X = np.load('C:/Users/Mriank Ghosh/Desktop/shell/dataset/AR_80m.npy')
# split dataset
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:len(X)]

with h5py.File('test_read.hdf5', 'w') as f:
	f.create_dataset('train', data = train)
	f.create_dataset('test', data = test)