# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:33:48 2021

@author: Beatrice Cantoni
"""
import scipy.stats as stats
import numpy as np

class Observation:
    def __init__(self, index, y, covs):
        self.index = index
        self.y = y
        self.covs = covs
    
    def likelihood(self, mean, variance):
        tmp = stats.lognorm.pdf(self.y, mean, variance**(1/2))
        if np.isnan(tmp) == True:
            tmp = 0
        return tmp    
    
    def pretty_print(self):
        print('observation', self.index)
        print('y', self.y)
        print('x', self.covs)

