#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:32:59 2020

@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def roc_curve(multi_labels, y_score, n_class):
    
    y_labels = np.where(multi_labels == n_class,1,0)
    print(y_labels.size)
    P = np.count_nonzero(y_labels)
    N = y_labels.size - P
    #threshold = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.96, 0.99, 0.9999, 0.999999, 1])
    threshold = np.array(range(0,1000))
    threshold = threshold /1000
    
    
    tpr = np.zeros(threshold.size)
    fpr = np.zeros(threshold.size)
    
    for i in range(threshold.size):
        for j in range(y_score.size):
            if(y_score[j] > threshold[i] and y_labels[j]==1):
                tpr[i] = tpr[i] + 1
            if(y_score[j] > threshold[i] and y_labels[j] == 0):
                fpr[i] = fpr[i] + 1
                
    tpr = tpr / P
    fpr = fpr / N

    
    roc = interpolate.interp1d(fpr,tpr,fill_value='extrapolate')
    x = np.arange(0,1,0.0001)
    y = roc(x)
    plt.plot(x,y)

        