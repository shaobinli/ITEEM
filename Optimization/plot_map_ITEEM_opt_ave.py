# -*- coding: utf-8 -*-
"""
1-13-2021
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose: plot optimization results
"""

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io
# load new modules developed for ITEEM
from Submodel_SWAT.SWAT_functions import get_yield, basic_landuse
from Submodel_SWAT.crop_yield import get_yield_crop


#  10 clusters 
cluster1 = [9,10,11,29,41]
cluster2 = [19,32,34,37]
cluster3 = [4,6,12]
cluster4 = [28,30,33]
cluster5 = [35,36,39,43,44,45]
cluster6 = [7,16,20,21,22,24,25,26,27,38,40,42]
cluster7 = [1,2,3]
cluster8 = [13,14,18,23]
cluster9 = [17]
cluster10 = [5,15]
clusters = [cluster1, cluster2, cluster3, cluster4, cluster5, 
            cluster6, cluster7, cluster8, cluster9, cluster10]

# load data again
# opt_X = scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_X_s2.mat')['out']
# opt_F_s2 = scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_F_s2.mat')['out']
# opt_F_s1 = scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_F_s1.mat')['out']
# opt_landuse = opt_X[0,:50].astype('float')
# evls_s2 = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\n_evals_s2.mat')['out']

# scenario = 'NSGA3_BMPs_Tech_biomass50_July23_2021'

def opt_landuse(scenario):
    # convert opt_landuse back to original landuse_matrix for get_yield() function
    opt_X = scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_X_'+scenario+'.mat')['out']
    # opt_F = scipy.io.loadmat('opt_F_'+scenario+'.mat')['out']
    landuse_matrix = np.zeros((opt_X.shape[0],45,62)) # BMP 37, 39, 46, 47, 48, 55
    for j in range(opt_X.shape[0]):
        opt_landuse = opt_X[j,:70].astype('float')    
        i = 0
        for cluster in clusters:
            for sw in cluster:
                # print(sw)
                landuse_matrix[j,sw-1, 1] = opt_landuse[i]
                landuse_matrix[j,sw-1,37] = opt_landuse[i+1]
                landuse_matrix[j,sw-1,39] = opt_landuse[i+2]
                landuse_matrix[j,sw-1,46] = opt_landuse[i+3]
                landuse_matrix[j,sw-1,47] = opt_landuse[i+4]
                landuse_matrix[j,sw-1,48] = opt_landuse[i+5]
                landuse_matrix[j,sw-1,55] = opt_landuse[i+6]
            i = i + 7
        landuse_matrix[j,7,1] = 1
        landuse_matrix[:,30,1] = 1   
        # opt_X[1,-1]==1
        if opt_X.shape[1]==74:
            landuse_matrix[j,7,:] = 0
            if opt_X[j,-1]==1 or opt_X[j,-2]==1 or opt_X[j,-3]==1:
                landuse_matrix[j,7,19] = 1.0  # assign BMP for subwatershed 8
            else: landuse_matrix[j,7,1] = 1.0 # 
               
        # landuse_matrix.sum(axis=1)
    return landuse_matrix

# landuse_matrix2 = opt_landuse('NSGA3_BMPs_Tech_biomass50_July23_2021')
# landuse_matrix_combinedS2 = opt_landuse('combinedS2_v2')
# landuse_matrix_BMPs_ET_S2_v2 = opt_landuse('BMPs_ET_S2_v2')
# landuse = landuse_matrix_BMPs_ET_S2_v2.mean(axis=0)
# landuse, land_agri = basic_landuse()

# plot the map
def plot_map_ave(name, scenario):
    '''month = {1,2,3..., 12}; 1 repersent January and so on.'''
    watershed_USRB = gpd.read_file(r'C:\ITEEM\Shapefiles\Watershed.shp')
    reach_USRB = gpd.read_file(r'C:\ITEEM\Shapefiles\reach.shp')
    
    landuse_matrix = opt_landuse(scenario).mean(axis=0)  # average of solutions
    # name='phosphorus'
    # select what type of outputs for plotting
    if name == 'nitrate' or name == 'phosphorus':
        yield_per_sw = get_yield(name, landuse_matrix)[1]
        unit = 'kg/ha'
        yield_per_sw_annual = yield_per_sw.sum(axis=1).mean(axis=0)
    elif name == 'sediment':
        yield_per_sw = get_yield(name, landuse_matrix)[1]
        unit = 'ton/ha'
        yield_per_sw_annual = yield_per_sw.sum(axis=1).mean(axis=0)
    if name == 'streamflow':
        yield_per_sw = get_yield(name, landuse_matrix)[1]
        unit = 'mm'
        yield_per_sw_annual = yield_per_sw.sum(axis=1).mean(axis=0)
    
    elif name == 'crop':
    # elif name == 'soybean' or name == 'corn' or name == 'corn sillage':
        yield_per_sw_corn = get_yield_crop('corn', landuse_matrix)[1]
        yield_per_sw_soy = get_yield_crop('soybean', landuse_matrix)[1]
        yield_per_sw = yield_per_sw_corn + yield_per_sw_soy
        unit = 'kg/ha'
        yield_per_sw_annual = yield_per_sw.mean(axis=0)
    
    '''baseline '''
    landuse_matrix_baseline = np.zeros((45,62))
    landuse_matrix_baseline[:,1] = 1
    if name == 'crop':
        yield_per_sw_baseline = get_yield_crop('corn', landuse_matrix_baseline)[1] + get_yield_crop('soybean', landuse_matrix_baseline)[1]
        yield_per_sw_annual_baseline = yield_per_sw_baseline.mean(axis=0)
    else:
        yield_per_sw_baseline = get_yield(name, landuse_matrix_baseline)[1]
        yield_per_sw_annual_baseline = yield_per_sw_baseline.sum(axis=1).mean(axis=0)

    reduction_prct = 100*(yield_per_sw_annual_baseline - yield_per_sw_annual)/yield_per_sw_annual_baseline
    reduction_prct = np.nan_to_num(reduction_prct)
    
    fig, ax = plt.subplots(figsize=(2,2))
    fig.tight_layout(pad=1)
    ax.set_axis_off()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    
    # yield_per_sw_df_month = yield_per_sw_df.iloc[:,j]
    yield_per_sw_annual_df = pd.DataFrame(reduction_prct)
    ## Using Merge on a column and Index in Pandas 
    watershed = watershed_USRB.merge(yield_per_sw_annual_df, left_index=True, right_index=True)
    
    # my_cmap = plt.cm.get_cmap('Pastel2_r')
    # my_cmap = plt.cm.get_cmap('viridis_r')
    # vmax= round(reduction_prct.max(), -1)
    if name == 'nitrate':
        vmax=60
    elif name == 'phosphorus':
        vmax=60
    elif name =='streamflow':
        vmax=30
    elif name=='sediment':
        vmax=80
    elif name == 'crop':
        vmax=30
        
    if vmax <=40:
        color_number = int(vmax/5)
        
    else: color_number = int(vmax/10)
    
    my_cmap = ListedColormap(sns.color_palette("viridis_r", color_number).as_hex())
    
    watershed.plot(column=0, legend=True, ax=ax, cmap=my_cmap,
                        edgecolor="grey", linewidth=0.5, 
                        cax=cax, vmin=0, vmax=vmax,
                        # legend=False
                        legend_kwds={'orientation': "horizontal", 'ticks': [i*vmax/color_number for i in range(color_number+1)]}
                        )
    reach_USRB.plot(ax=ax, color='blue', linewidth=0.8)
    
    # cax.set_yticklabels({'fontsize': 8})
    if name == 'nitrate':
        plt.text(x=0.5, y=0.02, fontsize=9, s= "N reduction (%)",
                  transform=ax.transAxes)
    elif name =='phosphorus':
        plt.text(x=0.5, y=0.02, fontsize=9, s= "P reduction (%)",
                  transform=ax.transAxes)        
    elif name == 'sediment':
        plt.text(x=0.55, y=0.02, fontsize=9, s="Sediment\nreduction (%)",
                  transform=ax.transAxes)
    elif name == 'streamflow':
        plt.text(x=0.55, y=0.02, fontsize=9, s="Streamflow reduction (%)",
                  transform=ax.transAxes)
    elif name == 'crop':
        plt.text(x=0.45, y=0.02, fontsize=9, s="Crop reduction (%)",
                  transform=ax.transAxes)    

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.0, wspace=0.0)
    # ax.set_axis_off()
    
    # fig.set_size_inches((2, 2), forward=False)
    # fig.savefig(r'C:\ITEEM\Optimization\figures\July2021\\' + str(name).capitalize() + '_' + scenario +'_Aug19.tif', 
    #             dpi=300, bbox_inches='tight')
    # fig.tight_layout(pad=1)
    # plt.tight_layout()
    # plt.show()
    return reduction_prct

# start = time.time()

# reduction_prct_BMPonly = plot_map_ave(name='crop', scenario='NSGA2_BMPsonly_biomass50_Sept13_2021')
# reduction_prct_BMPonly.mean()
# reduction_prct_BMPs_Tech = plot_map_ave(name='crop', scenario='NSGA2_BMPs_Tech_biomass50_Sept13_2021')
# reduction_prct_BMPs_Tech.mean()
# reduction_prct_BMPonly = plot_map_ave(name='phosphorus', scenario='NSGA2_BMPsonly_biomass50_Aug19_2021')
# reduction_prct_BMPonly = plot_map_ave(name='phosphorus', scenario='NSGA2_BMPs_Tech_biomass50_Aug19_2021')


# '''calculate weighted reduction at watershed scale'''
# from Submodel_SWAT.SWAT_functions import basic_landuse
# land_total, land_agri = basic_landuse()
# a=0
# for i in range(45):
#     a += reduction_prct_BMPonly[i]*land_agri[i,0]/land_agri.sum()
# print(a)

# end = time.time()
# print('Running time is {:.1f} seconds'.format(end-start))
# reduction_prct1 = plot_map_ave('nitrate', 'singleS2_v2')
# plot_map_ave('phosphorus', 'singleS2_v2')
# plot_map_ave('streamflow', 'singleS2_v2')
# plot_map_ave('sediment', 'singleS2_v2')

# scenario_list1 = ['singleS1_v2','singleS2_v2','combinedS1_v2', 'combinedS2_V2']
# scenario_list2 = ['BMPs_BT_S1_v2','BMPs_BT_S2_v2','BMPs_ET_S1_v2', 'BMPs_ET_S2_v2']

# for i in scenario_list2:
#     plot_map_ave('phosphorus', i)
#     plot_map_ave('streamflow', i)
#     plot_map_ave('sediment', i)
#     plot_map_ave('nitrate', i)


def bmp_prct(scenario):
    bmp = [37, 39, 46, 47, 48, 55]
    # scenario= 'NSGA2_BMPs_Tech_biomass100_June24_2021'
    landuse = opt_landuse(scenario).mean(axis=0) # average of solutions
    prct_list = []
    land_agri = basic_landuse()[1]
    for i in bmp:
         prct = np.matmul(landuse[:,i],np.array(land_agri))/land_agri.sum()
         print(prct)
         prct_list.append(prct[0])
    return prct_list

# scenario_list1 = ['singleS1_v2','singleS2_v2','combinedS1_v2', 'combinedS2_V2']
# scenario_list2 = ['BMPs_BT_S1_v2','BMPs_BT_S2_v2','BMPs_ET_S1_v2', 'BMPs_ET_S2_v2']
# bmp = bmp_prct('BMPs_ET_S2_v2')
# bmp = bmp_prct('NSGA2_BMPs_Tech_biomass100_June24_2021')
# opt_F = scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_F_singleS1_v2.mat')['out']
# opt_F = scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_F_BMPs_BT_S1_v2.mat')['out']


def stacked_bar_bygroup():
    '''plot BMP adoption fraction in each cluster'''
    opt_X_s1_df = pd.DataFrame(scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPsonly_biomass50_Aug19_2021.mat')['out'].mean(axis=0))*100
    opt_X_s2_df = pd.DataFrame(scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPs_Tech_biomass50_Aug19_2021.mat')['out'][:,:70].mean(axis=0))*100
    # opt_X_s2_df = pd.DataFrame(scipy.io.loadmat('C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPs_Tech_biomass100_June24_2021.mat')['out'][0,:60])*100 
    df = pd.concat([opt_X_s1_df, opt_X_s2_df])

    cluster_list = []
    for i in range(10):
        temp_list = ['Cluster' + str(i+1)]*7
        cluster_list.extend(temp_list)
    
    df['Clusters'] = cluster_list*2
    BMP_list = ['baseline','none_30%_FS', 'none_30%_GW', 'CC_30%_na', 
                'CC_30%_FS','CC_30%_GW', 'Perennial grass']
    
    df['BMP'] = BMP_list*20  
    df = df.reset_index()
    df.columns = ['index','value', 'Clusters', 'BMP']
    s1 = ['S1']*70
    s2 = ['S2']*70
    df['Scenario'] = s1+s2
    opt_X_s1_df['BMP'] = BMP_list*10
    opt_X_s1_df['Cluster'] = cluster_list
    opt_X_s2_df['BMP'] = BMP_list*10
    opt_X_s2_df['Cluster'] = cluster_list
    
    color = sns.color_palette("rocket_r",7)
    barWidth = 0.3
    edgecolor= 'black'
    # The position of the bars on the x-axis
    r1 = np.arange(10)
    r2 = [x + 0 + barWidth for x in r1]
    names = ['Cluster '+str(i+1) for i in range(10)]
    bar1_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[0]])][0]
    bar2_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[1]])][0]
    bar3_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[2]])][0]
    bar4_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[3]])][0]
    bar5_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[4]])][0]
    bar6_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[5]])][0]
    bar7_s1 = opt_X_s1_df[opt_X_s1_df['BMP'].isin([BMP_list[6]])][0]
    
    bar1_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[0]])][0]
    bar2_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[1]])][0]
    bar3_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[2]])][0]
    bar4_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[3]])][0]
    bar5_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[4]])][0]
    bar6_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[5]])][0]
    bar7_s2 = opt_X_s2_df[opt_X_s2_df['BMP'].isin([BMP_list[6]])][0]
    # make plot
    fig, ax= plt.subplots(figsize=(6.5,4))
    plt.bar(r1, bar1_s1, width=barWidth, color=color[0], edgecolor=edgecolor)
    plt.bar(r1, bar2_s1, bottom=bar1_s1, width=barWidth, color=color[1], edgecolor=edgecolor)
    plt.bar(r1, bar3_s1, bottom=(bar1_s1.reset_index() + bar2_s1.reset_index())[0], width=barWidth, 
            color=color[2], edgecolor=edgecolor)
    plt.bar(r1, bar4_s1, bottom=(bar1_s1.reset_index() + bar2_s1.reset_index()+
                             bar3_s1.reset_index())[0], width=barWidth, color=color[3], 
            edgecolor=edgecolor)
    plt.bar(r1, bar5_s1, bottom=(bar1_s1.reset_index() + bar2_s1.reset_index()+
                             bar3_s1.reset_index()+bar4_s1.reset_index())[0], width=barWidth,
            color=color[4], edgecolor=edgecolor)
    plt.bar(r1, bar6_s1, bottom=(bar1_s1.reset_index() + bar2_s1.reset_index()+
                             bar3_s1.reset_index()+bar4_s1.reset_index()+bar5_s1.reset_index())[0], width=barWidth,
            color=color[5], edgecolor=edgecolor)
    plt.bar(r1, bar7_s1, bottom=(bar1_s1.reset_index() + bar2_s1.reset_index()+
                             bar3_s1.reset_index()+bar4_s1.reset_index()+bar5_s1.reset_index()+bar6_s1.reset_index())[0], width=barWidth,
            color=color[5], edgecolor=edgecolor)
    
    plt.bar(r2, bar1_s2, width=barWidth, color=color[0], edgecolor=edgecolor)
    plt.bar(r2, bar2_s2, bottom=bar1_s2, width=barWidth, color=color[1], edgecolor=edgecolor)
    plt.bar(r2, bar3_s2, bottom=(bar1_s2.reset_index() + bar2_s2.reset_index())[0], width=barWidth, 
            color=color[2], edgecolor=edgecolor)
    plt.bar(r2, bar4_s2, bottom=(bar1_s2.reset_index() + bar2_s2.reset_index()+
                             bar3_s2.reset_index())[0], width=barWidth, 
            color=color[3], edgecolor=edgecolor)
    plt.bar(r2, bar5_s2, bottom=(bar1_s2.reset_index() + bar2_s2.reset_index()+
                             bar3_s2.reset_index()+bar4_s2.reset_index())[0], width=barWidth,
            color=color[4], edgecolor=edgecolor)
    plt.bar(r2, bar6_s2, bottom=(bar1_s2.reset_index() + bar2_s2.reset_index()+
                             bar3_s2.reset_index()+bar4_s2.reset_index()+bar5_s2.reset_index())[0], width=barWidth,
            color=color[5], edgecolor=edgecolor)
    plt.bar(r2, bar7_s2, bottom=(bar1_s2.reset_index() + bar2_s2.reset_index()+
                             bar3_s2.reset_index()+bar4_s2.reset_index()+bar5_s2.reset_index()+bar6_s2.reset_index())[0], width=barWidth,
            color=color[5], edgecolor=edgecolor)    
    
    plt.xticks((r1+r2)/2, names, rotation=30)
    # plt.xticks(r1, ['S1' for i in range(10)])
    # plt.xticks(r2, ['S2' for i in range(10)])
    plt.ylabel('% of agricultural land', fontsize=10)
    # plt.legend(BMP_list, loc='upper center', bbox_to_anchor=(1, 0.5),fontsize=11)
    plt.legend(BMP_list,loc='lower left', mode='expand', ncol=3, 
                     bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize=10)
    
    # plt.savefig(r'C:\ITEEM\Optimization\figures\July2021\NSGA3_Optimal_allocation_BMP_biomass50_Aug19.tif', 
    #             dpi=300, bbox_inches = 'tight')
    fig.tight_layout()
    # plt.xlabel("Clusters")

# stacked_bar_bygroup()
