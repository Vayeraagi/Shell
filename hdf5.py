# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:43:58 2022

@author: Mriank Ghosh
"""

# Python program to demonstrate
# HDF5 file

import numpy as np
import h5py

# initializing a random numpy array
arr = np.random.randn(1000)

# creating a file
with h5py.File('test.hdf5', 'w') as f:
	dset = f.create_dataset("default", data = arr)

hf = h5py.File('test.hdf5', 'r')
hf.keys()
n1 = hf.get('default')
train, test = n1[1:len(n1)-7], n1[len(n1)-7:]
######################################################################

import numpy as np
import h5py


arr1 = np.random.randn(10000)
arr2 = np.random.randn(10000)

with h5py.File('test_read.hdf5', 'w') as f:
	f.create_dataset('array_1', data = arr1)
	f.create_dataset('array_2', data = arr2)
    
with h5py.File('test_read.hdf5', 'r') as f:
	d1 = f['array_1']
	d2 = f['array_2']

	data = d2[d1[:]>1]


