
"""
Created on Feb 1 21:47:30 2021

@author: Beatrice Cantoni
"""

import numpy as np
from Pitman_yor.Classes.Grouped_obs_missing import Grouped_obs_missing
import settings


#%%
def calculate_mse_missing_PY(state, observation_index, 
                             alpha, sigma,
                             full_dataset,
                             prior_mu, prior_a):
    
    cluster_belong = state.observations_to_clusters[observation_index]
    
    possible_clusters = len(state.clusters_to_observations)+1
    clusters_weights_log = (-1)*np.ones((possible_clusters))
    clusters_predictions = (-1)*np.ones((possible_clusters))
    #calcolo dei cluster score
    for i in range(0, (possible_clusters-1)):        
        log_n_sigma = np.log(len(state.clusters_to_observations[i])-settings.sigma)
        
        #log_g
        state.add_to_cluster(observation_index, i) 
        log_g = Grouped_obs_missing(full_dataset, 
                                    state.clusters_to_observations, 
                                    i,
                                    prior_mu).log_grouped_g_missing(
                                        prior_mu, prior_a)
        state.remove_from_cluster(observation_index, i)
        
        #log prediction
        log_weight = log_n_sigma + log_g
        prediction = np.random.normal(state.clusters_parameters_mean[i], 
                                      state.clusters_parameters_variance[i]**(1/2))        
        clusters_predictions[i] = prediction
        clusters_weights_log[i] = log_weight
    
    #new cluster    
    log_alpha_sigma = np.log(alpha + sigma*(possible_clusters-1))
    state.add_cluster(observation_index) 
    
    
    log_g = Grouped_obs_missing(full_dataset, 
                                state.clusters_to_observations, 
                                i,
                                prior_mu).log_grouped_g_missing(prior_mu, 
                                                         prior_a)
    log_weight = log_alpha_sigma + log_g
    prediction = np.random.normal(state.clusters_parameters_mean[-1], 
                                      state.clusters_parameters_variance[-1]**(1/2)) 
    clusters_predictions[-1]= prediction
    state.remove_cluster(len(state.clusters)-1)
    #normalize #qui fare trick
    max_log_weight = np.max(clusters_weights_log)
    clusters_weights_log = clusters_weights_log - max_log_weight
    clusters_weights = np.exp(clusters_weights_log)
    clusters_weights_normalized = clusters_weights / np.sum(clusters_weights)
    
    #obtain weighted prediction
    weighted_predictions = clusters_weights_normalized*clusters_predictions
    state.predictions[observation_index] = np.sum(weighted_predictions)    
    
    state.observations_to_clusters[observation_index] = cluster_belong
    
    return state
#%%
def total_mse_missing_PY(state, full_data, alpha, sigma,
                      prior_mu, prior_a):
    for o in range(0, len(full_data)):
        state = calculate_mse_missing_PY(state, 
                                      o, 
                                      alpha, sigma,
                                      full_data,
                                      prior_mu, prior_a)
    
     
    return state




