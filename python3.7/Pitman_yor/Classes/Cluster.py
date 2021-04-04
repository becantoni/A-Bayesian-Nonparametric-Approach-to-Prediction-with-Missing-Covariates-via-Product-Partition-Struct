# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:39:21 2021

@author: Beatrice Cantoni
"""
import numpy as np
import settings

class Cluster:   
    def __init__(self, common_mean, common_variance, cluster_index):
        self.init_rnd_mean_variance(common_mean, common_variance)
        self.cluster_index = cluster_index
        
    def init_rnd_mean_variance(self, common_mean, common_variance): 
        self.mean = np.random.normal(common_mean, common_variance**(1/2), 1)
        self.variance = (np.random.uniform(0, settings.a_sigma, 1))
    
    def cluster_print(self):
        print('mean', self.mean)
        print('variance', self.variance)
    