# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:39:58 2019

@author: Shaobin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
#df = pd.read_excel('radar_chart.xlsx')

def plot_radar_single(row_index, single=True):
    df = pd.read_excel(r'C:\ITEEM\Optimization\radar_chart_July2021.xlsx', sheet_name='Sept', nrows=3)  
    ''' PART 1: Create background '''
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
    # scalar = MinMaxScaler()
    # df_scale = scalar.fit_transform(df.iloc[:,1:]).astype('float32')
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    #ax = plt.subplot(111, polar=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3),
                           subplot_kw=dict(polar=True))
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=12)
    ### Draw one axe per variable + add labels labels yet
    categories_v2 = ['Nitrate\n load', 
                     '    P\n   load', 
                      # ' Sediment\nlandscape\nloss',
                      ' Sediment\nload', 
                     '      Streamflow\noutlet', 
                     '      Energy\n  DWT', 
                     '  Energy\nGP',
                     'Energy\nWWT',
                     'Biomass',
                     '  Cost\nDWT', 
                       'Cost\nWWT', 
                     # 'Cost_GP', 
                     # 'Cost_crop', 
                     'Profit\nGP', 
                     'Profit\ncrop',
                     'Non-market\nbenefit',
                     'Total\nnet\nprofit', 
                     # 'recovered\nP_complex', 
                     'Corn\nproduction', 
                     'Soybean\nproduction',
                     'P\n recovery',]
    
    #Aligning rotated xtick labels with their respective xticks 
    #https://stackoverflow.com/questions/14852821/aligning-rotated-xticklabels-with-their-respective-xticks
    labels = ax.set_xticklabels(categories_v2)
    for i, label in enumerate(labels):
        label.set_y(label.get_position()[0] - 0.1)
    
    ax.set_xticklabels(labels=categories_v2, position=(10,0))
    pos=ax.get_rlabel_position()
    ax.set_rlabel_position(pos+7)
    
    # Draw ylabels (scales)
    ax.set_rlabel_position(7)
    ax.tick_params(axis='y', labelsize=6)
    # plt.yticks(np.arange(0,1), ['0.2', '0.4', '0.6', '0.8', '1.0'],
    #            color='black', size=22)
    # plt.ylim(0,1)
    plt.yticks([1,2,3,4], ["0","0.33","0.66","1"], color="black", size=8)
    plt.ylim(0,4)
    # ax.set_yticklabels(['0', '0.25', '0.50', '1.0'])
    ax.set_xticklabels([])
    ax.xaxis.grid(True,color='grey',linestyle='-')
    ax.yaxis.grid(True,color='k',linestyle='-')
    
    # fill color
    i = 5
    k = [i,i,i]
    color=['lightblue', 'goldenrod', 'red', 'green']
    color=['w', 'w', 'w', 'w']
    
    bars = plt.bar(angles[0:1], k, width=(2*np.pi/N), bottom=0.0, color=color[0], alpha = 0.3)
    bars = plt.bar(angles[1:2], k, width=(2*np.pi/N), bottom=0.0, color=color[0], alpha = 0.3)
    bars = plt.bar(angles[2:3], k, width=(2*np.pi/N), bottom=0.0, color=color[0], alpha = 0.3)
    bars = plt.bar(angles[3:4], k, width=(2*np.pi/N), bottom=0.0, color=color[0], alpha = 0.3)
    bars = plt.bar(angles[4:5], k, width=(2*np.pi/N), bottom=0.0, color=color[1], alpha = 0.2)
    bars = plt.bar(angles[5:6], k, width=(2*np.pi/N), bottom=0.0, color=color[1], alpha = 0.2)
    bars = plt.bar(angles[6:7], k, width=(2*np.pi/N), bottom=0.0, color=color[1], alpha = 0.2)
    bars = plt.bar(angles[7:8], k, width=(2*np.pi/N), bottom=0.0, color=color[2], alpha = 0.15)
    bars = plt.bar(angles[8:9], k, width=(2*np.pi/N), bottom=0.0, color=color[2], alpha = 0.15)
    bars = plt.bar(angles[9:10], k, width=(2*np.pi/N), bottom=0.0, color=color[2], alpha = 0.15)
    bars = plt.bar(angles[10:11], k, width=(2*np.pi/N), bottom=0.0, color=color[2], alpha = 0.15)
    bars = plt.bar(angles[11:12], k, width=(2*np.pi/N), bottom=0.0, color=color[2], alpha = 0.15)
    bars = plt.bar(angles[12:13], k, width=(2*np.pi/N), bottom=0.0, color=color[2], alpha = 0.15)
    # bars = plt.bar(angles[13:14], k, width=(2*np.pi/N), bottom=0.0, color='red', alpha = 0.15)
    # bars = plt.bar(angles[14:15], k, width=(2*np.pi/N), bottom=0.0, color='red', alpha = 0.15)
    bars = plt.bar(angles[13:14], k, width=(2*np.pi/N), bottom=0.0, color=color[3], alpha = 0.15)
    bars = plt.bar(angles[14:15], k, width=(2*np.pi/N), bottom=0.0, color=color[3], alpha = 0.15)
    bars = plt.bar(angles[15:16], k, width=(2*np.pi/N), bottom=0.0, color=color[3], alpha = 0.15)
    bars = plt.bar(angles[16:17], k, width=(2*np.pi/N), bottom=0.0, color=color[3], alpha = 0.15)
    
    # bars = plt.bar(angles[6.5:7], [5,5,5], width=(2*np.pi/N), bottom=0.0,color='goldenrod',align='edge')
    
    ''' PART 2: Add plots '''
    # S0
    color=[sns.color_palette("husl", 9)]
    color=['black', 'green', 'blue']
    
    if single == True:
        values=df.loc[0].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, marker='s', linestyle='dashed', label="Baseline", color=color[0])
        
        values=df.loc[row_index].drop('group').values.flatten().tolist()
        # values = df_scale[0].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, marker='s', linestyle='solid', label="Baseline", color=color[row_index])
        # ax.fill(angles, values, 'b', alpha=0.1)
        # Add legend
        # plt.legend(loc='upper center', bbox_to_anchor=(0.9, 1.2), frameon=False, prop={'size': 12})
    
    elif single ==False:
        for i in range(3):  # 9 scenarios, including baseline
            values=df.loc[i].drop('group').values.flatten().tolist()
            # values = df_scale[0].tolist()
            values += values[:1]
            if i == 0:
                line = 'dashed'
            else: line = 'solid'
            
            ax.plot(angles, values, linewidth=1.5, marker='s', linestyle=line, label="Baseline", color=color[i])
            row_index = '_all'
            
    plt.tight_layout()
    # plt.savefig(r'C:\ITEEM\ITEEM_figures\Polar_chart_ITEEM_Oct_normalized(2-4)v2.tif',dpi=150)
    plt.savefig(r'C:\ITEEM\Optimization\figures\July2021\spider_plot_row_index'+str(row_index)+'Sept23_2021.tif',dpi=150)
    # plt.savefig(r'C:\ITEEM\Optimization\figures\spider_plot_blank.pdf')

# for i in range(3):
#     plot_radar_single(i)
plot_radar_single(2, single=False)
