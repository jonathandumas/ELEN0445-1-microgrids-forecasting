# -*- coding: UTF-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import process_dad_data, build_random_LS_VS, point_scores, plot_point_forecasts, plot_point_metrics

# ------------------------------------------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------------------------------------------

k1 = 11 # 0 or 11
k2 = 80 # 95 or 80
output_dim = k2 - k1 + 1
model_name = 'persistance' #
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
# ------------------------------------------------------------------------------------------------------------------
# FIXME: start of students part
# Build the list of previous days, get the corresponding values, and reindex, compute prediction on VS
# WARNING !!! predictions must be of shape (VS_days, output_dim)

day_list = ... # TODO
df_predictions = ... # TODO


# FIXME: end of student part
# ------------------------------------------------------------------------------------------------------------------

df_predictions.index = df_VS_targets.index
df_predictions.columns = df_VS_targets.columns


# Create folder
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
