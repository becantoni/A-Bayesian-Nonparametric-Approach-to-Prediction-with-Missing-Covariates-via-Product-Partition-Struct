# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:19:20 2021

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
        tmp = stats.norm.pdf(self.y, mean, variance**(1/2))
        log_lik = np.log(tmp)
        return log_lik    
    
    def pretty_print(self):
        print('observation', self.index)
        print('y', self.y)
        print('x', self.covs)

