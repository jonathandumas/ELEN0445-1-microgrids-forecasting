# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    k1 = 11  # 0 or 11
    k2 = 80  # 95 or 80
    model_list = ['persistance', 'MLR', 'MLP', 'GBR']

    # ------------------------------------------------------------------------------------------------------------------
    # Load scores
    # ------------------------------------------------------------------------------------------------------------------
    df_scores = dict()
    # NMAE and NRMSE scores
    for model_name in model_list:
        dirname = 'export/' + model_name + '/scores/'
        df_scores[model_name] = pd.read_csv(dirname + 'point_scores_' + model_name + '_' + str(k1) + '_' + str(k2) + '.csv', index_col=0)

    # CRPS scores
    # FIXME comment the next line when only considering point forecasts
    df_scores['GBR_CRPS'] = pd.read_csv(dirname + 'quantile_scores_GBR_' + str(k1) + '_' + str(k2) + '.csv', index_col=0)

    for model_name in model_list:
        print('%s model %.1f NMAE %.1f NRMSE' % (model_name, df_scores[model_name].mean()['NMAE'], df_scores[model_name].mean()['NRMSE']))
    # FIXME comment the next line when only considering point forecasts
    print('GBR model CRPS %.1f' % (df_scores['GBR_CRPS'].mean()['CRPS']))

    FONTSIZE = 20
    plt.figure()
    plt.plot(df_scores['persistance'].index, df_scores['persistance']['NMAE'].values, linewidth=3, color='blue',label='persistance')
    plt.plot(df_scores['MLR'].index, df_scores['MLR']['NMAE'].values, linewidth=3, color='red',label='MLR')
    plt.plot(df_scores['MLP'].index, df_scores['MLP']['NMAE'].values, linewidth=3, color='k',label='MLP')
    plt.plot(df_scores['GBR'].index, df_scores['GBR']['NMAE'].values, linewidth=3, color='green',label='GBR')
    # FIXME comment the next line when only considering point forecasts
    plt.plot(df_scores['GBR_CRPS'].index, df_scores['GBR_CRPS'].values, '--g', linewidth=3,label='GBR quantile')
    plt.ylim(0, 30)
    plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig('export/comparison_nmae.pdf')
    plt.close('all')

    FONTSIZE = 20
    plt.figure()
    plt.plot(df_scores['persistance'].index, df_scores['persistance']['NRMSE'].values, '--b', linewidth=3,label='persistance')
    plt.plot(df_scores['MLR'].index, df_scores['MLR']['NRMSE'].values, '--r', linewidth=3,label='MLR')
    plt.plot(df_scores['MLP'].index, df_scores['MLP']['NRMSE'].values, '--k', linewidth=3, label='MLP')
    plt.plot(df_scores['GBR'].index, df_scores['GBR']['NRMSE'].values, '--g', linewidth=3, label='GBR')
    plt.ylim(0, 30)
    plt.ylabel('%', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig('export/comparison_nrmse.pdf')
    plt.close('all')