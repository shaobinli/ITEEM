# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:46:17 2021

@author: Shaobin
"""

'''plot'''
import numpy as np
from pymoo.visualization.pcp import PCP
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

'''get data'''
opt_F_BMPs_Tech_biomass50 = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\opt_F_NSGA2_BMPs_Tech_biomass50_Sept13_2021_DWT.mat')['out']
opt_F_BMPonly_biomass50 = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\opt_F_NSGA2_BMPsonly_biomass50_Sept13_2021.mat')['out']

opt_F_combined = np.concatenate((opt_F_BMPonly_biomass50, opt_F_BMPs_Tech_biomass50), axis=0)

scalar = MinMaxScaler()
opt_F_combined_rescaled = scalar.fit_transform(opt_F_combined)
opt_F_combined_rescaled[:,[3, 4]] = opt_F_combined_rescaled[:,[4, 3]]
opt_F_combined_rescaled = np.delete(opt_F_combined_rescaled, 4, axis=1)
# data_scaled_df2 = data_scaled_df2.reindex([0,1,4,3,2])

'''pcp plot pandas parallel_coordinates'''
from pandas.plotting import parallel_coordinates
# df_final = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/diamonds_filter.csv")
df = pd.DataFrame(opt_F_combined_rescaled)
df.columns = ["Water\nquality", "Crop\nproduction", "System\nbenefit", 'P\nrecovery']#, "Energy\ndemand"]
df['class'] = 'BMPs only'
df.iloc[100:,-1] = 'BMPs + EBTs'

tech_wwt_list = ['_AS', '_ASCP', '_EBPR', '_EBPR-A', '_EBPR-S']
tech_GP_list = ['1', '2', '3']
opt_X_BMP_Tech = scipy.io.loadmat(r'C:\ITEEM\Optimization\solutions\opt_X_NSGA2_BMPs_Tech_biomass50_Sept13_2021.mat')['out']
i=0
for i in range(100):
    df.iloc[100+i,-1] = 'BMPs + EBTs' + tech_wwt_list[int(opt_X_BMP_Tech[i,70])]
    # if opt_X_BMP_Tech[i,71] + opt_X_BMP_Tech[i,72] + opt_X_BMP_Tech[i,73] == 0:
    #     df.iloc[100+i,-1] = 'BMPs + EBTs' + tech_wwt_list[int(opt_X_BMP_Tech[i,70])]
    # elif opt_X_BMP_Tech[i,71] + opt_X_BMP_Tech[i,72] + opt_X_BMP_Tech[i,73] == 1:
    #     df.iloc[100+i,-1] = 'BMPs + EBTs' + tech_wwt_list[int(opt_X_BMP_Tech[i,70])] + '_1'
    # elif opt_X_BMP_Tech[i,71] + opt_X_BMP_Tech[i,72] + opt_X_BMP_Tech[i,73] == 2:
    #     df.iloc[100+i,-1] = 'BMPs + EBTs' + tech_wwt_list[int(opt_X_BMP_Tech[i,70])] + '_2'
    # elif opt_X_BMP_Tech[i,71] + opt_X_BMP_Tech[i,72] + opt_X_BMP_Tech[i,73] == 2:
    #     df.iloc[100+i,-1] = 'BMPs + EBTs' + tech_wwt_list[int(opt_X_BMP_Tech[i,70])] + '_3'
        
# for i in range(100):
#     df.iloc[100+i,-1] = 'BMPs + EBTs' + tech_wwt_list[int(opt_X_BMP_Tech[i,70])]
df.groupby('class').count()


fig, ax = plt.subplots(figsize=(6.5, 3))
color = ['blue', 'green', 'y', 'red', 'dimgrey']  # ['lime', 'aquamarine]
# color = ['cornflowerblue', 'lightcoral', 'red','maroon']
import seaborn as sns
# color = sns.color_palette("rocket", 11)
# blue_rgb = (0,0,255)
# color = [blue_rgb, color[1]]

parallel_coordinates(df, 'class', color=color, linewidth=1, alpha = 0.8)
handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

ax.legend(loc='center left', bbox_to_anchor=(-0.015, 1.15), ncol=2, prop={'family':'Arial', 'size':10})
# plt.ylabel('Normalized value [0,1]', fontdict={'family':'Arial', 'size':10})
plt.xticks(fontname="Arial", fontsize=10)
# plt.yticks(np.arang0, 1.1, 0.25)
# plt.yticks(fontsize=10)
# plt.ylim(-0.1,1.1)
# plt.grid(False)
# plt.savefig(r'C:\ITEEM\Optimization\figures\July2021\plot_pcp_Oct_2021_color2.tif', 
#             dpi=300, bbox_inches = 'tight')
plt.show()



# '''pcp plot from pymoo'''
# plot = PCP(
#             figsize=(6.5,4),
#             # title=("Run", {'pad': 30}),
#             n_ticks=10,
#             legend=(True, {'bbox_to_anchor': (0.75,1.15), 'ncol':2}),
#             # legend=(True, {'loc': 'best'}),
#             labels=["Water\nquality",  
#                     "Crop\nproduction",
#                     "Economic\nbenefit",
#                     'P\nrecovery',
#                     "Energy\ndemand", 
#                     ], 
#             # ylim=(0,2)
#             # tight_layout=True,
#             # fontsize2=10
#             )

# plot.set_axis_style(color="grey", alpha=1)


# plot.add(opt_F_combined_rescaled[:100], color="blue", alpha=0.3)
# plot.add(opt_F_combined_rescaled[100:opt_F_BMPs_Tech_biomass50.shape[0]+opt_F_BMPonly_biomass50.shape[0]],
#          color="green", alpha=0.3)
# plot.add(opt_F_combined_rescaled[0], color="blue", alpha=0.5, label='BMPs only')
# plot.add(opt_F_combined_rescaled[opt_F_BMPonly_biomass50.shape[0]], color="green", alpha=0.5, label='BMPs + Engineering Techs')


# # plot.add(opt_F_combined_rescaled[200:300], color="red", alpha=0.3)
# # plot.add(opt_F_combined_rescaled[300:400], color="grey", alpha=0.3)
# # plot.add(opt_F_combined_rescaled[200], color="red", alpha=0.5, label='BMPs only (biomass100)')
# # plot.add(opt_F_combined_rescaled[300], color="grey", alpha=0.5, label='BMPs + Engineering Techs (biomass100)')

# # plot.add(opt_F_combined_rescaled[400:499], color="black", alpha=0.3)
# # plot.add(opt_F_combined_rescaled[500:599], color="purple", alpha=0.3)
# # plot.add(opt_F_combined_rescaled[400], color="purple", alpha=0.5, label='BMPs only (biomass150)')
# # plot.add(opt_F_combined_rescaled[500], color="black", alpha=1, label='BMPs + Engineering Techs (biomass150)')

# # plot.reset()
# # pyplot.ylim((0.0, 1.00))

# # plt.ylim(0,1)
# # plot.normalize_each_axis = False
# plot.bounds=[[0,0,0,0,0],[1,1,1,1,1]]
# # plt.legend('')

# # plot.save(r'C:\ITEEM\Optimization\figures\July2021\plot_pcp_July28.tif', dpi=300)
# plot.show()
