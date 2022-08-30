# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:34:15 2021

@author: Shaobin

Purpose: get spider output
"""

from Optimization.plot_map_ITEEM_opt_ave import opt_landuse
from ITEEM import ITEEM 
import scipy.io
import pandas as pd
import numpy as np


'''BMPs_Tech_biomass50'''
opt_X = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPs_Tech_biomass50_June2021.mat')['out']
landuse_matrix = opt_landuse('NSGA2_BMPs_Tech_biomass50_June2021')

df_spider_output2 = pd.DataFrame()
df_obj2 = pd.DataFrame()
tech_wwt_list = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']
tech_GP_list = [int(1), int(2)]
landuse_matrix_BMP_tech = opt_landuse('NSGA2')

for i in range(100):
    wwt_tech = opt_X[:,60].astype('int')
    gp_tech1 = opt_X[:,61].astype('int')
    gp_tech2 = opt_X[:,62].astype('int')
    gp_tech3 = opt_X[:,63].astype('int')
    output = ITEEM(landuse_matrix_BMP_tech[i,:,:], tech_wwt=tech_wwt_list[wwt_tech[i]], limit_N=10.0, 
                    tech_GP1=tech_GP_list[gp_tech1[i]], 
                    tech_GP2=tech_GP_list[gp_tech2[i]], 
                    tech_GP3=tech_GP_list[gp_tech3[i]])
    run_ITEEM_opt = output.run_ITEEM_opt()
    df_spider_output2['run'+str(1+i)] = run_ITEEM_opt[-1]
    df_obj2['run'+str(1+i)] = [run_ITEEM_opt[0], run_ITEEM_opt[1], run_ITEEM_opt[2], run_ITEEM_opt[3]]


'''BMPs only_biomass50'''
landuse_matrix_BMPonly = opt_landuse('NSGA2_BMPonly_biomass50_June2021')
df_spider_output = pd.DataFrame()
df_obj = pd.DataFrame()

for i in range(100):
    output = ITEEM(landuse_matrix_BMPonly[i,:,:], tech_wwt='AS', limit_N=10.0, tech_GP1=1, tech_GP2=1, tech_GP3=1)
    run_ITEEM_opt = output.run_ITEEM_opt()
    df_spider_output['run'+str(1+i)] = run_ITEEM_opt[-1]
    df_obj['run'+str(1+i)] = [run_ITEEM_opt[0], run_ITEEM_opt[1], run_ITEEM_opt[2], run_ITEEM_opt[3]] 
