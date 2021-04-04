# -*- coding: utf-8 -*-
"""
Created on Feb 1 21:47:30 2021

@author: Beatrice Cantoni
"""
#%%
from Pitman_yor.training_pitman_yor import algo_missing_PY
from Pitman_yor.training_pitman_yor import chinese_restaurant_PY
from Pitman_yor.dataset_pitman_yor import split_dataset_PY
from Pitman_yor.dataset_pitman_yor import convert_to_list
from Pitman_yor.dataset_pitman_yor import convert_to_dataframe
import settings

#%%
TRIALS = 1
data_imported #import data as dataframe
data_missing = convert_to_list(data_imported)  
df_m = convert_to_dataframe(data_missing)

TRAINING_ITERATIONS = 200

mse_total = []
mse_total_DP = []

for i in range(0,TRIALS):
    full_dataset = data_missing.copy()
    num_covariates = len(full_dataset[0].covs)
    split = split_dataset_PY(full_dataset)
    training_data = split[0]
    test_data = split[1]
    full_data = split[2]
    
    prior_variance = [0.5]*(num_covariates)
    prior_a = [1 / x for x in prior_variance]
    prior_mu = [1]*(num_covariates)
    
    S0 = chinese_restaurant_PY(
        training_data,
        full_data,
        settings.alpha)
    
    States = [S0]
       
    results = algo_missing_PY(
        TRAINING_ITERATIONS,
        training_data,
        full_data,
        States,
        prior_mu, 
        prior_a)
     
    predictions = results
    final_prediction = [sum(x) / len(x) for x in zip(*predictions)]
    real_values = []
    for o in range(0, len(full_data)):
        real_values.append(full_data[o].y)
    mse = [x - y for x, y in zip(final_prediction, real_values)]
    mse_sqrt = [x**2 for x in mse]

    