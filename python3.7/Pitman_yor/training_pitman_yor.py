# -*- coding: utf-8 -*-
"""
Created on Feb 1 21:47:30 2021

@author: Beatrice Cantoni
"""


from Pitman_yor.Classes.State import State
from Pitman_yor.Classes.Grouped_obs_missing import Grouped_obs_missing
from Pitman_yor.test_mse import total_mse_missing_PY
import random
import numpy as np
import settings

#%%

def chinese_restaurant_PY(observations, 
                          observations_full,
                          alpha):
    restaurant = State(settings.observations_mean, settings.v, 
                       settings.a_sigma_0, 
                       len(observations), 
                       len(observations_full))
    restaurant.add_cluster(0)
    table_sizes = [1]
    all_tables = [0]
    for i in range(1, len(observations)):
        p_new_table = (alpha + (max(all_tables)*settings.sigma))/(alpha + i)
        if (random.random()<p_new_table):
            restaurant.add_cluster(i) 
            table_sizes.append(1)
            all_tables.append(len(all_tables))
        else:
            weights = [x - settings.sigma for x in table_sizes]           
            weights = [x / (alpha + i) for x in weights]
            table = np.random.choice(
                all_tables,
                1,
                weights
                )[0]
            restaurant.add_to_cluster(i,table)
            table_sizes[table]+=1
    return restaurant


#%%
def parameters_update_missing_PY(state, dataset):
    state.cluster_mean_update(dataset)
    state.common_mean_update()
    state.cluster_variance_update(settings.mh_variance, dataset)
    state.common_variance_update(settings.mh_variance)
    return state


#%%
def Neal8_missing_PY(state, observation_index,
                  dataset,
                  prior_mu, prior_a):  
    
    cluster_from = state.observations_to_clusters[observation_index]
    
    
    if len(state.clusters_to_observations[cluster_from]) == 1:
        state.remove_cluster(cluster_from)
        state.rescale_observations_to_clusters()
    else:
        state.remove_from_cluster(observation_index, cluster_from) 

    
    clusters_score = (-1)*(np.ones(len(state.clusters_to_observations)+1))
    for i in range(0, (len(clusters_score)-1)):
        log_n_s = np.log(len(state.clusters_to_observations[i])-settings.sigma)
        log_likelihood = dataset[observation_index].likelihood(
            state.clusters_parameters_mean[i], 
            state.clusters_parameters_variance[i])
        state.add_to_cluster(observation_index, i) 
        log_g = Grouped_obs_missing(dataset, 
                                    state.clusters_to_observations, 
                                    i).log_grouped_g_missing()
        state.remove_from_cluster(observation_index, i)
        
        cluster_score = log_n_s + log_g + log_likelihood
        clusters_score[i]= cluster_score
    
    #new cluster
    log_alpha_s = np.log(settings.alpha + len(state.clusters_to_observations)*settings.sigma)
    state.add_cluster(observation_index) 
    log_likelihood = dataset[observation_index].likelihood(
        state.clusters_parameters_mean[len(state.clusters)-1], 
        state.clusters_parameters_variance[len(state.clusters)-1])
    log_g = Grouped_obs_missing(dataset, 
                                state.clusters_to_observations, 
                                (len(state.clusters)-1)
                                ).log_grouped_g_missing()
    new_score = log_alpha_s + log_g + log_likelihood 
    
    clusters_score[-1]= new_score
    
    #normalize
    max_score=np.max(clusters_score)
    clusters_score = clusters_score - max_score
    clusters_score = np.exp(clusters_score)
    normalized_scores = clusters_score / np.sum(clusters_score)
    #select cluster
    selected_cluster = np.random.choice(
        np.arange(0, (len(state.clusters))), 
        p=normalized_scores)    
    
    #remove new cluster if not assigned
    if selected_cluster < len(state.clusters)-1:
        state.remove_cluster(len(state.clusters)-1)
        state.add_to_cluster(observation_index, 
                             selected_cluster)
    
    return state    



#%%
def total_Neal_missing_PY(state, dataset,
                       prior_mu, prior_a):
    for o in range(0, len(dataset)):
        state = Neal8_missing_PY(state, o,
                              dataset,
                              prior_mu, 
                              prior_a)
    return state


#%%

def algo_missing_PY(iterations, dataset_training, full_data,
                 States,
                 prior_mu, prior_a):
    state = States[0]    
    predictions=[]    
    for i in range(0,iterations):
        state = total_Neal_missing_PY(state, dataset_training,
                                   prior_mu, prior_a)
        state = parameters_update_missing_PY(state, dataset_training)
        state = total_mse_missing_PY(state, 
                                     full_data, 
                                     settings.alpha,
                                     settings.sigma,
                                     prior_mu, 
                                     prior_a)        
        predictions.append(state.predictions)

    return predictions 
       
                                                                     

