# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose: validation test of response matrix method for SWAT
Only works for USRW considering point source and reservoir trapping
"""

# Import required packages for data processing
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from Submodel_SWAT.SWAT_functions import basic_landuse, response_mat, landuse_mat
from Submodel_SWAT.results_validation import get_yield, loading_per_sw


def loading_outlet_USRW(name, scenario_name):
    '''
    reservoir watershed: 33; downstream of res: 32
    outlet: 34
    '''
    df = pd.read_excel(r'C:\ITEEM\Submodel_SWAT\results_validation\NitrateAndStreamflowAtSub32.xlsx', sheet_name=2)
    df[np.isnan(df)] = 0
    loading_BMP_sum = loading_per_sw(name, scenario_name)
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[1], loading_BMP_sum.shape[2]))
    
    for i in range(33):
        a = df.loc[i].unique().astype('int')
        a = a[a!=0]
        for j in a:
            # print (j)
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]
            
    # Total loading in sw32 = res_out + background loading
    res_in = outlet[:,:,32]
    res_in_cum = np.zeros((16,12))
    for i in range(16):
        res_in_cum[i,:] = np.cumsum(res_in)[i*12: (i+1)*12]
    if name == 'nitrate':
        res_out_cum = res_in_cum * 0.725  # assuming 27.5% trapping efficiency for nitrate
    elif name =='phosphorus':
        res_out_cum = res_in_cum * 0.443  # assuming 55.7% trapping efficiency for nitrate
    elif name =='phosphorus':
        res_out_cum = res_in_cum * 0.129  # assuming 87.1% trapping efficiency for nitrate  
    elif name =='streamflow':
        res_out_cum = res_in_cum * 0.926  # assuming 7.4% trapping efficiency for nitrate
        
    res_out_cum_flatten = res_out_cum.flatten()
    res_out = np.zeros((192,1))
    res_out[0] = res_out_cum_flatten[0]
    res_out[1:192,0] = np.diff(res_out_cum_flatten)
    res_out2 = np.zeros((16,12))
    for i in range(16):
        res_out2[i,:] = res_out[i*12: (i+1)*12].T
    # sw32 is the downstream of reservoir
    outlet[:,:,31] = loading_BMP_sum[:,:,31] + res_out2
    
    # add point source
    # Calculate loading in sw31 with point source
    df2_point=0
    # loading_BMP_sum[i,j,30] = ANN...
    if name =='nitrate':
        # point_Nitrate = 1315.43*30 # kg/month, average
        outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
    elif name == 'phosphorus':
        # point_TP = 1923.33*30# kg/month, average
        outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
        
    # b contains all upstream subwatersheds for sw31
    b = df.loc[30].unique().astype('int')
    b = b[b!=0]
    
    # get unique subwatersheds that do not contribute to reservoir
    for i in range(33,45):
        c = df.loc[i].unique().astype('int')
        c = c[c!=0]
        d = list(set(c) - set(b))
        # Key step: the following equation takes the trapping efficiency into account. 
        # All upstream contributions of sw32 is reconsidered with trapping efficiency 
        if 31 in list(c):
            # print ('true, i=',i)
            outlet[:,:,i] = outlet[:,:,30]
        for j in d:
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]
    # update the loadings for upperstream that has higher values
    e = b[b>33] 
    for i in e:
        f = df.loc[i-1].unique().astype('int')
        f = f[f!=0]
        for j in f:
            outlet[:,:,i-1] += loading_BMP_sum[:,:,j-1]
    return outlet

# test_N = loading_outlet_USRW('nitrate', 'Sheet1')
# test_N_1D = test_N.flatten()
# test_N_run2 = loading_outlet_USRW('nitrate', 'Sheet2')
# test_N_1D_run2 = test_N_run2.flatten()

# test_TP = loading_outlet_USRW('phosphorus', 'Sheet1')
# test_TP_1D = test_TP.flatten()
# test_TP_run2 = loading_outlet_USRW('phosphorus', 'Sheet2')
# test_TP_1D_run2 = test_TP_run2.flatten()