# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:36:02 2022

@author: Mriank Ghosh
"""

from memory_profiler import profile

@profile
def main_func():
    import random
    arr1 = [random.randint(1,10) for i in range(100000)]
    arr2 = [random.randint(1,10) for i in range(100000)]
    arr3 = [arr1[i]+arr2[i] for i in range(100000)]
    del arr1
    del arr2
    tot = sum(arr3)
    del arr3
    print(tot)

main_func()
