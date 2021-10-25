# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load weather forecasts
    Tp = pd.read_csv("data/TT2M_MAR_dad_12.csv", parse_dates=True, index_col=0)  # MAR GFS -> forecasts
    Tp.columns = ['Tp' + str(i) for i in range(len(Tp.columns))]
    Ip = pd.read_csv("data/SWD_MAR_dad_12.csv", parse_dates=True, index_col=0)  # MAR GFS -> forecasts
    Ip.columns = ['Ip' + str(i) for i in range(len(Ip.columns))]
    Tp.index = Tp.index.tz_localize('UTC')  # set weather data index to UTC timezone
    Ip.index = Ip.index.tz_localize('UTC')  # set weather data index to UTC timezone

    Tp_reshape = Tp.values.reshape(Tp.shape[0] * Tp.shape[1])
    Tp_reshape = pd.DataFrame(data=Tp_reshape, index=pd.date_range(start=Tp.index[0], periods=Tp.shape[0] * Tp.shape[1], freq='15T'), columns=['Tp']).truncate(before=pd.Timestamp('2020-04-11'),
                                                                   after=pd.Timestamp('2020-09-14'))

    Ip_reshape = Ip.values.reshape(Ip.shape[0] * Ip.shape[1])
    Ip_reshape = pd.DataFrame(data=Ip_reshape, index=pd.date_range(start=Ip.index[0], periods=Ip.shape[0] * Ip.shape[1], freq='15T'), columns=['Ip']).truncate(before=pd.Timestamp('2020-04-11'),
                                                                   after=pd.Timestamp('2020-09-14'))

    # Load pv data
    df_pv = pd.read_csv('data/PV_uliege_parking_power_15min.csv', parse_dates=True, index_col=0)['power_total'].to_frame().truncate(before=pd.Timestamp('2020-04-11'),
                                                                   after=pd.Timestamp('2020-09-14'))
    df_pv.columns = ['Pm']

    # PLOT
    plt.figure()
    Tp_reshape.plot()
    plt.show()

    plt.figure()
    Ip_reshape.plot()
    plt.show()

    plt.figure()
    df_pv.plot()
    plt.show()

    df_concat = pd.concat([Ip_reshape, df_pv], axis=1, join="inner")

