# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:29:47 2022

@author: Mriank Ghosh
"""

import numpy as np
import h5py

X = np.load('C:/Users/Mriank Ghosh/Desktop/shell/dataset/MA_8m.npy')
# split dataset
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:len(X)]

with h5py.File('8.hdf5', 'w') as f:
	f.create_dataset('train', data = train)
	f.create_dataset('test', data = test)
