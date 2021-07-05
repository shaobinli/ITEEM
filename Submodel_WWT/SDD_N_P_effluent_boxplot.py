# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose: cleanning and ploting for SDD influent data for monthy data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_excel(r'C:\ITEEM\Submodel_WWT\SDD_N_P_1989_2020.xlsx', parse_dates=['Date'],index_col='Date')
# df = pd.read_excel(r'C:\ITEEM\Submodel_WWT\SDD_N_P_2012-2019.xlsx', parse_dates=['Date'],index_col='Date', usecols='A:L')


def box_data_SDD():
    # df = pd.read_excel(r'C:\ITEEM\Submodel_WWT\SDD_N_P_2012-2019.xlsx', parse_dates=['Date'],
    #                     index_col='Date', usecols='A:L')
    df = pd.read_excel(r'C:\ITEEM\Submodel_WWT\SDD_N_P_1989_2020.xlsx', parse_dates=['Date'],
                        index_col='Date')
    start_date = '2012-01-01'  # '2012-01-01'
    end_date = '2019-12-31'    # '2019-12-31'
    df[df.columns[5]] = df[df.columns[5]].replace('<0.12', '0.12')
    df[df.columns[6]] = df[df.columns[6]].replace('<', '0.12')
    # setting errors=’coerce’,transform the non-numeric values into NaN.
    df[df.columns[5]] = pd.to_numeric(df[df.columns[5]], errors='coerce') 
    mask = (df.index > start_date) & (df.index <= end_date)
    df = df.loc[mask]
    df = df.dropna(subset=[df.columns[5], df.columns[6], df.columns[8], df.columns[10]])
    
    # df_organicN = df.dropna(subset=[df.columns[7]])
    #add month and week columns
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week 
    df['year'] = df.index.year
    #remove week of 53
    df = df[df.week != 53]
    # remove redundant columns
    df2 = df.iloc[:, 5:]
    df2.columns = ['NH3', 'Nitrate', 'Organic_N','TP', 'Effluent flow (MGD)', 'Effluent flow (m3/d)',
                   'Nitrate_loading', 'TP_loading', 'month', 'week', 'year'] 
    df2['TN_loading'] = (df2['NH3'] + df2['Organic_N']+ df2['Nitrate'])*df2[df2.columns[5]]/1000
    df2['TN'] = df2['NH3'] + df2['Nitrate'] + df2['Organic_N']
    df2['tech'] = 'SDD'
    df2 = df2.reset_index(drop=True)
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    week_list = ['Week_' + str(i) for i in range(1,53)]

    df4 = pd.DataFrame()
    pd.options.mode.chained_assignment = None  # remove warning signal; default='warn'
    for i in range(12):
            df3 = df2[df2.month==i+1]
            df3.iloc[:,8] = month_list[i]
            df4 = pd.concat([df4,df3])
    df5 = pd.DataFrame()
    for i in range(52):
            df3 = df4[df4.week==i+1]
            df3.loc[:,'week'] = week_list[i]
            df5 = pd.concat([df5,df3])
    return df5

# data_SDD = box_data_SDD()
# data_SDD['Nitrate'].mean()
# data_SDD['TN'].mean()

def box_plot_SDD(name, period, unit):
    '''
    name: ['TP', 'Nitrate']
    period: ['week', 'month']
    unit: ['concentration', 'loading']
    '''
    df2 = box_data_SDD()
    x = df2[period]
    y = df2[name]
    plt.figure(figsize=(6,4))
    sns.boxplot(x=x, y=y, palette="Set1", showfliers = False)
    if unit =='concentration':
        plt.ylabel( name +' effluent concentration (mg/L)',fontsize=14)
    elif unit =='loading':
        plt.ylabel( name +' effluent loading (kg/day)',fontsize=14)
  
    plt.xticks(fontsize=11, rotation=90)
    plt.xlabel('Time interval' , fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(False)
    # leg = plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5),fontsize=14)
    plt.savefig(r'C:\ITEEM\Submodel_WWT\SDD_analysis\figures\Dec2020\'' + name + unit + '.tif', 
                dpi=300, bbox_inches = 'tight')
    plt.tight_layout()
    plt.show()

# box_plot_SDD('TP_loading', 'month', 'loading')
# box_plot_SDD('Nitrate', 'month', 'loading')
# box_plot_SDD('TN', 'month', 'concentration')


