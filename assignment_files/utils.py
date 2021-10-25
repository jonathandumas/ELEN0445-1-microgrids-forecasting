# -*- coding: utf-8 -*-

import time
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def scale_data(df_data_ls:pd.DataFrame, df_data_vs:pd.DataFrame):
    """
    1. Scale data using the StandardScaler on the learning set.
    2. Transform the inputs of the learning and valiation sets.
    :param df_data_ls: pd.DataFrame with the learning set data to fit the scaler.
    :param df_data_vs: pd.DataFrame with the validation set data.
    :return the learning and validation scaled data.
    """

    scaler = StandardScaler()
    scaler.fit(df_data_ls)
    return scaler.transform(df_data_ls), scaler.transform(df_data_vs)


def process_dad_data(k1: int=0, k2: int=95, quantile: bool = False, N_Q: int = 9):
    """
    Process input data for day ahead forecasts:
    - load weather forecasts
    - load pv observations
    :param k1: forecasting time steps k is between k1 and k2.
    :param k2: forecasting time steps k is between k1 and k2.
    :param quantile: day ahead point or quantile forecasts.
    :param N_Q: number of quantile to forecast.
    """
    # Load weather forecasts
    Tp = pd.read_csv("data/TT2M_MAR_dad_12.csv", parse_dates=True, index_col=0)  # MAR GFS -> forecasts
    Tp.columns = ['Tp' + str(i) for i in range(len(Tp.columns))]
    Ip = pd.read_csv("data/SWD_MAR_dad_12.csv", parse_dates=True, index_col=0)  # MAR GFS -> forecasts
    Ip.columns = ['Ip' + str(i) for i in range(len(Ip.columns))]
    Tp.index = Tp.index.tz_localize('UTC')  # set weather data index to UTC timezone
    Ip.index = Ip.index.tz_localize('UTC')  # set weather data index to UTC timezone

    # Load pv data
    df_pv = pd.read_csv('data/PV_uliege_parking_power_15min.csv', parse_dates=True, index_col=0)['power_total'].to_frame()
    df_pv.columns = ['Pm']
    df_pv_reshaped = pd.DataFrame(data=df_pv.values.reshape(int(df_pv.shape[0] / 96), 96),
                                  index=[day for day in df_pv.resample('D').mean().dropna().index])

    # Build weather data
    Tp = Tp[Tp.columns[k1:k2 + 1]]
    Ip = Ip[Ip.columns[k1:k2 + 1]]
    # shape = [nb_days, 2*(k2-k1+1)]
    df_inputs = pd.concat([Tp, Ip], axis=1, join='inner').truncate(before=pd.Timestamp('2020-04-11'),
                                                                   after=pd.Timestamp('2020-09-14'))
    # Build target for point forecasts
    # shape = [nb_days, k2-k1+1]
    df_target_pv = df_pv_reshaped[df_pv_reshaped.columns[k1:k2 + 1]]
    df_target_pv = df_target_pv.truncate(before=pd.Timestamp('2020-04-11'), after=pd.Timestamp('2020-09-14'))

    # Build target for quantile forecasts with shape = [nb_days, N_q*(k2-k1+1)]
    if quantile:
        target_list = []
        for col in df_target_pv.columns:
            target_list += [df_target_pv[col]] * N_Q

        # shape = [nb_days, N_Q*(k2-k1+1)]
        df_target_pv = pd.concat(target_list, axis=1, join='inner')
        col_list = []
        for i in range(k1, k2 + 1):
            col_list += ['q' + str(i) + '_' + str(j) for j in range(1, N_Q + 1)]
        df_target_pv.columns = col_list

    return df_inputs, df_target_pv


def build_random_LS_VS(df_inputs: pd.DataFrame, df_target_pv: pd.DataFrame, VS_days: int, random_state: int):
    """
    Build a random pair Learning Set, Validation Set.
    :param df_inputs: inputs.
    :param df_target_pv: targets.
    :param VS_days: Validation Set size.
    :param random_state: parameter to build randomly the pair.
    """
    df_VS_inputs = df_inputs.sample(n=VS_days, random_state=random_state)
    df_VS_inputs = df_VS_inputs.sort_index()
    df_LS_inputs = df_inputs.drop(df_VS_inputs.index)

    df_LS_targets = df_target_pv.drop(df_VS_inputs.index)
    df_VS_targets = df_target_pv.drop(df_LS_targets.index)

    return df_VS_inputs, df_LS_inputs, df_LS_targets, df_VS_targets


def point_scores(y_true:np.array, y_pred:np.array, k1:int, k2:int):
    """
    Compute NMAE and NRMSE.
    ---------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs) -> n_outputs = nb of forecasting periods.
    Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs) -> n_outputs = nb of forecasting periods.
    Estimated target values.

    k1 : first forecasting period. -> 0 <= k1 <= 95
    k2 : last forecasting period. -> 0 <= k2 <= 95 and k1 <= k2.

    Returns
    -------
    pd.DataFrame with the NMAE and NRMSE per forecasting time period.
    """

    nmae = 100 * mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')  / 466.4
    nrmse = 100 * np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values'))  / 466.4
    df_scores = pd.DataFrame(data=nmae, columns=['NMAE'])
    df_scores['NRMSE'] = nrmse
    df_scores.index = [i for i in range(k1, k2 + 1)]

    return df_scores


def crps_nrg_k(y_true:float, y_quantiles:np.array):
    """
    Compute the CRPS NRG for a given leadtime k.
    :param y_true: true value for this leadtime.
    :param y_quantiles: quantile predictions for this leadtime with shape(N_Q,)
    """
    N_Q = y_quantiles.shape[0] # Nb of quantiles predicted.
    simple_sum = np.sum(np.abs(y_quantiles - y_true)) / N_Q
    double_somme = 0
    for i in range(N_Q):
        for j in range(N_Q):
            double_somme += np.abs(y_quantiles[i] - y_quantiles[j])
    double_sum = double_somme / (2 * N_Q * N_Q)

    crps = simple_sum  - double_sum

    return crps

def crps_over_vs(df_pred:pd.DataFrame, df_true:pd.DataFrame, output_dim:int, N_Q:int, k1:int, k2:int):
    """
    Compute the average of the CRPS over the validation set.
    :param df_pred: quantile predictions (n_days, nb_quantiles * nb_forecasting_periods).
    :param df_true: targets (n_days, nb_forecasting_periods)..
    :param output_dim: nb_forecasting_periods.
    :param N_Q: number of quantiles.
    :param k1: first forecasting period. -> 0 <= k1 <= 95
    :param k2: last forecasting period. -> 0 <= k2 <= 95 and k1 <= k2.
    """

    crps_list = []
    for leadtime in range(0, output_dim):
        crps_k = 0
        for day in df_pred.index:
            df_forecasts_dad_day = df_pred.loc[day].values.reshape(output_dim, N_Q)
            crps_k += crps_nrg_k(y_true=df_true.loc[day].iloc[leadtime], y_quantiles=df_forecasts_dad_day[leadtime,:])
        crps_k = crps_k / len(df_pred.index)
        crps_list.append(crps_k)
    df_crps = pd.DataFrame(data=crps_list, index=[i for i in range(k1, k2+1)], columns=['CRPS']) / 466.4 * 100

    return df_crps


def plot_point_forecasts(df_predictions:pd.DataFrame, df_target:pd.DataFrame, dir:str, model_name:str, k1: int=0, k2: int=95):

    x_index = [i for i in range(k1, k2+1)]
    for day in df_predictions.index:
        FONTSIZE = 20
        plt.figure()
        plt.plot(x_index, df_target.loc[day].values, 'r', linewidth=3, label='Pm')
        plt.plot(x_index, df_predictions.loc[day].values, 'k', linewidth=3, label='Pp')
        plt.ylim(0, 466.4)
        plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(dir + 'point_' + day.strftime('%Y%m%d') + '_' + model_name + '_' + str(k1) + '_' + str(k2) + '.pdf')
        plt.close('all')


def plot_point_metrics(df_scores: pd.DataFrame, dir: str, model_name: str, k1: int = 0, k2: int = 95):
    FONTSIZE = 20
    plt.figure()
    plt.plot(df_scores.index, df_scores['NMAE'].values, linewidth=3, color='b', label='NMAE')
    plt.plot(df_scores.index, df_scores['NRMSE'].values, linewidth=3, color='r', label='NRMSE')
    plt.ylim(0, 25)
    plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dir + 'point_scores_' + model_name + '_' + str(k1) + '_' + str(k2) + '.pdf')
    plt.close('all')

if __name__ == "__main__":
    print('ok')
