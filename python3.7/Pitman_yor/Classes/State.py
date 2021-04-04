# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:46:08 2021

@author: Beatrice Cantoni
"""
import numpy as np
from Pitman_yor.Classes.Cluster import Cluster
import settings


class State:
    def __init__(self, m_0, v, 
                 a_sigma_0,
                 num_observations, 
                 num_full_observations):
        self.clusters = []
        self.clusters_to_observations = []
        self.observations_to_clusters = [-1]*num_full_observations
        self.predictions = [-1]*num_full_observations
        self.errors = [-1]*num_full_observations
        self.clusters_parameters_mean = []
        self.clusters_parameters_variance = []
        self.common_mean = 0
        self.common_variance = 100 
        self.m_0 = m_0 
        self.a_sigma_0 = a_sigma_0 
        self.v = v 
        
        
    def add_cluster(self, observation_index):
        self.clusters.append(Cluster(self.common_mean, self.common_variance, len(self.clusters)))#qui senza -1 perchè lo stai creando
        self.clusters_to_observations.append([observation_index])
        self.observations_to_clusters[observation_index] = len(self.clusters)-1#così stai settando il cluster index #qui -1 perchè è già stato creato
        self.clusters_parameters_mean.append(self.clusters[len(self.clusters)-1].mean)#??
        self.clusters_parameters_variance.append(self.clusters[len(self.clusters)-1].variance)
    
    def add_to_cluster(self, observation_index, cluster_index):
        self.clusters_to_observations[cluster_index].append(observation_index)
        self.observations_to_clusters[observation_index] = cluster_index
        
    def remove_from_cluster(self, observation_index, cluster_index):
        self.clusters_to_observations[cluster_index].remove(observation_index)

    def rescale_observations_to_clusters(self):
        for c in range(0, len(self.clusters)):            
            for item in self.clusters_to_observations[c]:
                self.observations_to_clusters[item]=c
                
    def remove_cluster(self, cluster_index):
        self.clusters_parameters_mean.pop(cluster_index)
        self.clusters_parameters_variance.pop(cluster_index)
        self.clusters.pop(cluster_index)
        self.clusters_to_observations.pop(cluster_index)#qui forse devo mettere index
    
    def common_variance_update(self, mh_variance):        
        new_proposal = np.random.normal(self.common_variance, (mh_variance**(1/2)))
        mean_dev =[]
        if new_proposal > 50:    
            for c in range(0,len(self.clusters)):
                mean_dev.append((
                    self.clusters_parameters_mean[c] - self.common_mean)**2)
            suff_stat = np.sum(mean_dev)/2        
            log_ratio = len(self.clusters)/2 * np.log(
                self.common_mean/new_proposal) * (
                    suff_stat * (
                        1/self.common_variance)- 1/new_proposal)            
            log_u = np.log(np.random.uniform())        
            if log_ratio > log_u:
                self.common_variance = new_proposal

    def cluster_variance_update(self, mh_variance, dataset):
        for j in range(0, len(self.clusters_to_observations)):
            new_proposal =  np.random.normal(self.clusters_parameters_variance[j], (mh_variance**(1/2)))
            if new_proposal < 20 and new_proposal > 0:   
                #print('old proposal',self.clusters_parameters_variance[j], 'new proposal', new_proposal)
                mean_dev =[]
                for item in self.clusters_to_observations[j]:
                    mean_dev.append((dataset[item].y - self.clusters_parameters_mean[j])**2)
                suff_stat = np.sum(mean_dev)/2        
                
                log_ratio = (len(self.clusters_to_observations[j])/2) * np.log(
                    self.clusters_parameters_variance[j]/new_proposal) + (
                        suff_stat * (1/self.clusters_parameters_variance[j])-1/new_proposal)            
                log_u = np.log(np.random.uniform())        
                if log_ratio > log_u:
                    self.clusters_parameters_variance[j] = new_proposal
    def common_mean_update(self):
        conditional_mean = (self.v**2 * (
            np.sum(self.clusters_parameters_mean)) + (
                self.common_variance*self.m_0))/(
                    self.v**2 * len(self.clusters) + self.common_variance)
        conditional_variance = (self.v**2 * self.common_variance)/(self.v**2 * len(self.clusters) + self.common_variance)
        self.common_mean = np.random.normal(conditional_mean, (conditional_variance**(1/2)))
        
    def cluster_mean_update(self, dataset):
        for j in range(0, len(self.clusters_to_observations)):
            sum_x=[]
            for item in self.clusters_to_observations[j]:
                sum_x.append(dataset[item].y)
            suff_stat = np.sum(sum_x) 
            conditional_mean = (self.common_variance * suff_stat + (self.clusters_parameters_variance[j]*self.common_mean))\
                /(self.common_variance * len(self.clusters_to_observations[j]) + self.clusters_parameters_variance[j])
            conditional_variance = (self.common_variance * self.clusters_parameters_variance[j])/(
                self.common_variance * len(self.clusters_to_observations[j]) + self.clusters_parameters_variance[j])           
            self.clusters_parameters_mean[j]= np.random.normal(conditional_mean, conditional_variance**(1/2))
            
    def pretty_print(self):
        print('c2o', self.clusters_to_observations)
        print('o2c', self.observations_to_clusters)
        