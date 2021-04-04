# -*- coding: utf-8 -*-
"""
Created on Feb 1 21:47:30 2021

@author: Beatrice Cantoni
"""

import numpy as np
import scipy.special as sc
import math as math    

#%% normal gamma univariate
    
def posterior_elements_normal_gamma(prior_mu, 
                                    prior_kappa, 
                                    prior_a, 
                                    prior_b, 
                                    n_j,
                                    mean,
                                    mean_deviance,
                                    prior_deviance):
    posterior_kappa = (prior_kappa + n_j)
    posterior_a = prior_a + (n_j/2)
    posterior_mu = (prior_kappa*prior_mu + n_j*mean)/(prior_kappa +n_j)
    posterior_beta = prior_b + (1/2)*mean_deviance + (prior_kappa*n_j*prior_deviance)/(2*(posterior_kappa))
    
    return posterior_mu, posterior_kappa, posterior_a, posterior_beta

def g_marginal_log_normal_gamma(prior_mu, 
                                prior_kappa, 
                                prior_a, 
                                prior_b,
                                posterior_mu, 
                                posterior_kappa, 
                                posterior_a, 
                                posterior_b,
                                n_j):
    g1=sc.loggamma(posterior_a)
    g2=sc.loggamma(prior_a)
    g3=prior_a*np.log(prior_b)
    g4=posterior_a*np.log(posterior_b)
    g5=(1/2)*np.log(prior_kappa/posterior_kappa)
    g6=-(n_j / 2)*np.log(2*math.pi)
    
    g=g1-g2+g3-g4+g5+g6
    return g


                     
    

    
    