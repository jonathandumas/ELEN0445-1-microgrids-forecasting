# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from utils import process_dad_data, build_random_LS_VS, scale_data, crps_over_vs

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

k1 = 11 # 0 or 11
k2 = 80 # 95 or 80
output_dim = k2 - k1 + 1
model_name = 'GBR' #
n_estimators = 50
max_depth = 5
learning_rate = 1e-2
VS_days = 15
N_Q = 9 # Number of quantiles
q_set = np.array([i / (N_Q+1) for i in range(1, N_Q + 1)]) # Set of quantiles

# ------------------------------------------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------------------------------------------
df_inputs, df_target_pv = process_dad_data(k1=k1, k2=k2, quantile=True, N_Q=N_Q)
df_target_pv_reshaped = pd.DataFrame(data=df_target_pv.values[:,::N_Q], index=df_target_pv.index, columns=[i for i in range(k1, k2 + 1)])

# ------------------------------------------------------------------------------------------------------------------
# DAD FORECASTER
# ------------------------------------------------------------------------------------------------------------------
print('%s model k1 = %s k2 = %s n_estimators = %s lr = %s max_depth = %s' % (model_name, k1, k2, n_estimators, learning_rate, max_depth))

# Build a random pair of LS/VS
df_VS_inputs, df_LS_inputs, df_LS_targets, df_VS_targets = build_random_LS_VS(df_inputs=df_inputs,
                                                                              df_target_pv=df_target_pv, VS_days=15,
                                                                              random_state=0)

# Reshape for GBR from (nb_days, 2*output_dim) to (nb_days * output_dim, 2)
df_input_LS_reshaped = pd.concat([pd.DataFrame(data=df_LS_inputs.values[:, :output_dim].reshape(df_LS_inputs.shape[0] * output_dim)), pd.DataFrame(data=df_LS_inputs.values[:, output_dim:].reshape(df_LS_inputs.shape[0] * output_dim))], axis=1, join='inner')
df_input_VS_reshaped = pd.concat([pd.DataFrame(data=df_VS_inputs.values[:, :output_dim].reshape(df_VS_inputs.shape[0] * output_dim)), pd.DataFrame(data=df_VS_inputs.values[:, output_dim:].reshape(df_VS_inputs.shape[0] * output_dim))], axis=1, join='inner')

# Reshape for GBR from (nb_days, N_q * output_dim) to (nb_days * output_dim,)
df_LS_targets_reshaped = df_LS_targets.values[:, ::N_Q].reshape(df_LS_targets.shape[0] * output_dim)
df_VS_targets_reshaped = df_VS_targets.values[:, ::N_Q].reshape(df_VS_targets.shape[0] * output_dim)

# ------------------------------------------------------------------------------------------------------------------
# FIXME: start of student part
# Scale the inputs, implement the model, fit it, and reshape predictions

# Scale the inputs
# TODO

# Training parameters
model_gbr = dict()
for quantile in q_set:
    print('Fit quantile %s' %(quantile))
    # TODO

    # Fit model
    # TODO

# Predictions
predictions = dict()
for quantile in q_set:
    predictions[quantile] = ... # TODO

# FIXME: end of student part
# ------------------------------------------------------------------------------------------------------------------
# Rebuild final predictions
predictions = pd.DataFrame(data=predictions).values.reshape(df_VS_inputs.shape[0],output_dim*N_Q)
df_predictions = pd.DataFrame(data=predictions, index=df_VS_targets.index, columns=df_VS_targets.columns)

# Create folder
dirname = 'export/'+model_name+'/forecasts/'
if not os.path.isdir(dirname):  # test if directory exist
    os.makedirs(dirname)
df_predictions.to_csv(dirname + 'dad_quantile_' + model_name + '_' + str(k1) + '_' + str(k2) + '.csv')


# ------------------------------------------------------------------------------------------------------------------
# COMPUTE CRPS
# ------------------------------------------------------------------------------------------------------------------
df_crps = crps_over_vs(df_pred=df_predictions, df_true=df_target_pv_reshaped, output_dim=output_dim, N_Q=N_Q, k1=k1, k2=k2)
print(df_crps.mean())

dirname = 'export/'+model_name+'/scores/'
if not os.path.isdir(dirname):  # test if directory exist
    os.makedirs(dirname)
df_crps.to_csv(dirname + 'quantile_scores_' + model_name + '_' + str(k1) + '_' + str(k2) + '.csv')

FONTSIZE = 20
plt.figure()
plt.plot(df_crps.index, df_crps.values, linewidth=3, color='b', label='CRPS')
plt.ylim(0, 25)
plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig(dirname + 'quantile_scores_' + model_name + '_' + str(k1) + '_' + str(k2) + '.pdf')
plt.close('all')


# ------------------------------------------------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------------------------------------------------

df_point = pd.read_csv('export/'+model_name+'/forecasts/' + 'dad_point_' + model_name + '_' + str(k1) + '_' + str(k2) + '.csv', parse_dates=True, index_col=0)

# Create folder
dirname = 'export/'+model_name+'/figures/'
if not os.path.isdir(dirname):  # test if directory exist
    os.makedirs(dirname)

x_index = [i for i in range(k1, k2+1)]
for day in df_predictions.index:
    df_forecasts_day = pd.DataFrame(data=df_predictions.loc[day].values.reshape(output_dim,N_Q), index=[i for i in range(k1, k2+1)])
    FONTSIZE = 20
    plt.figure()
    for j in range(1, N_Q // 2):
        plt.fill_between(x_index, df_forecasts_day[j + N_Q // 2].values, df_forecasts_day[(N_Q // 2) - j].values, alpha=0.5 / j, color=(1 / j, 0, 1))
    plt.plot(x_index, df_forecasts_day[0].values, 'b', linewidth=3, label='$q_1=$' + str(q_set[0]))
    plt.plot(x_index, df_forecasts_day[N_Q-1].values, 'b', linewidth=3, label='$q_Q=$' + str(q_set[N_Q - 1]))
    plt.plot(x_index, df_target_pv_reshaped.loc[day].values, 'r', linewidth=3, label='Pm')
    plt.plot(x_index, df_point.loc[day].values, 'k', linewidth=3, label='Pp')
    plt.ylim(0, 500)
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + 'quantile_' + day.strftime('%Y%m%d') + '_' + model_name + '_' + str(k1) + '_' + str(k2) + '.pdf')
    plt.close('all')


