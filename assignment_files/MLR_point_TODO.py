# -*- coding: UTF-8 -*-

import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils import process_dad_data, build_random_LS_VS, scale_data, point_scores, plot_point_forecasts, \
    plot_point_metrics

# ------------------------------------------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------------------------------------------

k1 = 11 # 0 or 11
k2 = 80 # 95 or 80
output_dim = k2 - k1 + 1
model_name = 'MLR' #
VS_days = 15

# ------------------------------------------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------------------------------------------
df_inputs, df_target_pv = process_dad_data(k1=k1, k2=k2)

# ------------------------------------------------------------------------------------------------------------------
# DAD FORECASTER
# ------------------------------------------------------------------------------------------------------------------

print('%s model k1 = %s k2 = %s' % (model_name, k1, k2))

# Build a random pair of LS/VS
df_VS_inputs, df_LS_inputs, df_LS_targets, df_VS_targets = build_random_LS_VS(df_inputs=df_inputs,
                                                                              df_target_pv=df_target_pv, VS_days=VS_days,
                                                                              random_state=0)

# MLR inputs shape from (nb_days, 2*output_dim) to (nb_days * output_dim, 2)
df_input_LS_reshaped = pd.concat([pd.DataFrame(data=df_LS_inputs.values[:, :output_dim].reshape(df_LS_inputs.shape[0] * output_dim)), pd.DataFrame(data=df_LS_inputs.values[:, output_dim:].reshape(df_LS_inputs.shape[0] * output_dim))], axis=1, join='inner')
df_input_VS_reshaped = pd.concat([pd.DataFrame(data=df_VS_inputs.values[:, :output_dim].reshape(df_VS_inputs.shape[0] * output_dim)), pd.DataFrame(data=df_VS_inputs.values[:, output_dim:].reshape(df_VS_inputs.shape[0] * output_dim))], axis=1, join='inner')

#####################
# Single output model
####################
# MLR targets shape from (nb_days, output_dim) to (nb_days * output_dim,)
df_LS_targets_reshaped = df_LS_targets.values.reshape(df_LS_targets.shape[0] * output_dim)
df_VS_targets_reshaped = df_VS_targets.values.reshape(df_VS_targets.shape[0] * output_dim)

# ------------------------------------------------------------------------------------------------------------------
# FIXME: start of student part
# Scale the inputs, implement the model, fit it, and reshape predictions
# TODO

# Build MLR model
# TODO

# Fit model
# TODO

# Compute Predictions on VS
# WARNING !!! predictions must be of shape (VS_days, output_dim) -> reshape the predictions
predictions = ... # TODO

# FIXME: end of student part
# ------------------------------------------------------------------------------------------------------------------

df_predictions = pd.DataFrame(data=predictions, index=df_VS_targets.index, columns=df_VS_targets.columns).sort_index()

# Create folder to save predictions
dirname = 'export/'+model_name+'/forecasts/'
if not os.path.isdir(dirname):  # test if directory exist
    os.makedirs(dirname)

df_predictions.to_csv(dirname + 'dad_point_' + model_name + '_' + str(k1) + '_' + str(k2) + '.csv')

# ------------------------------------------------------------------------------------------------------------------
# COMPUTE NMAE and NRMSE
# ------------------------------------------------------------------------------------------------------------------
df_scores = point_scores(y_true=df_VS_targets.values, y_pred=df_predictions.values, k1=k1, k2=k2)
print(df_scores.mean())

dirname = 'export/'+model_name+'/scores/'
if not os.path.isdir(dirname):  # test if directory exist
    os.makedirs(dirname)
df_scores.to_csv(dirname + 'point_scores_' + model_name + '_' + str(k1) + '_' + str(k2) + '.csv')

plot_point_metrics(df_scores=df_scores, dir=dirname, model_name=model_name, k1=k1, k2=k2)


# ------------------------------------------------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------------------------------------------------

# Create folder
dirname = 'export/'+model_name+'/figures/'
if not os.path.isdir(dirname):  # test if directory exist
    os.makedirs(dirname)

plot_point_forecasts(df_predictions=df_predictions, df_target=df_VS_targets, dir=dirname, model_name=model_name, k1=k1, k2=k2)