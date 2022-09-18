# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:39:43 2022

@author: Mriank Ghosh
"""

from memory_profiler import profile
@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()
