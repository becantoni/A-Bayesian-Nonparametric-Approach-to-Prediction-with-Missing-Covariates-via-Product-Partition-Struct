"""
Created on Sun Feb  7 14:33:48 2021

@author: Beatrice Cantoni
"""

import numpy as np
from distributions import posterior_elements_normal_gamma
from distributions import g_marginal_log_normal_gamma
import settings
import pandas as pd

class Grouped_obs_missing:
    def __init__(self, dataset, c2o, cluster_index): 
        self.group_obs(dataset, c2o, cluster_index)
        self.get_elements(dataset)
    
    def group_obs(self, dataset, c2o, cluster_index):
        group_obs = []
        for i in c2o[cluster_index]:
            group_obs.append(dataset[i])
        self.grouped_obs = group_obs

    def get_elements(self,dataset):
        
        means=[]
        for l in range(0, len(dataset[0].covs)):
             group_sum=[]
             for i in range(0, len(self.grouped_obs)):
                 if pd.isna(self.grouped_obs[i].covs[l]) == False:
                     group_sum.append(self.grouped_obs[i].covs[l])
             means.append(np.sum(group_sum)/len(self.grouped_obs))
        self.means = means
        
        mean_deviations_squared=[]
        for l in range(0, len(dataset[0].covs)):
             group_mean_devs=[]
             for i in range(0, len(self.grouped_obs)):
                 if pd.isna(self.grouped_obs[i].covs[l]) == False:
                     group_mean_devs.append((self.grouped_obs[i].covs[l]-means[l])**2)
             mean_deviations_squared.append(np.sum(group_mean_devs))
        self.means_deviation_squared = mean_deviations_squared
        
        
        dev_from_prior_squared = []
        for l in range(0, len(dataset[0].covs)):
            dev_from_prior_squared.append((means[l]-settings.covariates_mean)**2)
        self.dev_from_prior_squared = dev_from_prior_squared 
        
        
    def log_grouped_g_missing(self): #entrambi sono vettori
        n_j=len(self.grouped_obs)
        single_covariate_g_log =[]
        for l in range(0, len(self.grouped_obs[0].covs)):
            elements = posterior_elements_normal_gamma(
                                        settings.covariates_mean, 
                                        settings.prior_kappa, 
                                        settings.covariates_variance, 
                                        settings.prior_b, 
                                        n_j,
                                        self.means[l],
                                        self.means_deviation_squared[l],
                                        self.dev_from_prior_squared[l])
            
            g_log_missing_single = g_marginal_log_normal_gamma(
                                    settings.covariates_mean, 
                                    settings.prior_kappa, 
                                    settings.covariates_variance, 
                                    settings.prior_b,
                                    elements[0], 
                                    elements[1], 
                                    elements[2], 
                                    elements[3],
                                    n_j)
            single_covariate_g_log.append(g_log_missing_single)
            g_log_missing = np.sum(g_log_missing_single)
            
        return g_log_missing
    
          
                  
    










            