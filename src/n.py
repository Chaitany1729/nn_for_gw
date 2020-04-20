#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:22:41 2020

@author: chaitanya
"""

import h5py
import numpy
import random
import waveformGenerator

for i in range(5000):
    
    param = dict()
    
    m1 = random.randint(10, 60)
    param.update('m1', m1)
    m2 = random.randint(10, 60)
    param.update('m2', m1)
    SNR = random.randomint(1000, 3000)/100
    param.update('snr', m1)
    
    
    
    
    
    

