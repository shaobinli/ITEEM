# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:23:58 2020

@author: Shaobin
"""

# Import required packages for data processing
import pandas as pd
from pandas import ExcelFile
from pandas import ExcelWriter
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

# Import machine learning packages
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Import Hyper-parameter selection and tuning
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# color = sns.color_palette("rocket", 7)
color = sns.color_palette("husl", 7)


fig, ([ax1, ax2], [ax5, ax6]) = plt.subplots(2, 2, figsize=(6.5,5), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.0, wspace=0.0)
# ax1
data = pd.read_csv('C:\ITEEM\Submodel_WWT\simulation_data_csv\simulation_10000runs_AS_Dec.csv', skiprows=range(1,2))
data = data[(data.iloc[:,17]>data.iloc[:,2]*0.0052*0.99) & (data.iloc[:,17]<data.iloc[:,2]*0.0052*1.01)]
xdata = data.iloc[:,2:6]
ydata = data.iloc[:,9:17]
scalar = MinMaxScaler()
xdata = scalar.fit_transform(xdata).astype('float32')
ydata = scalar.fit_transform(ydata).astype('float32')
# Step 3: Define variables
Nop_var = len(xdata[0])
Noutput_var = len(ydata[0])
Ns = len(xdata[:])
Ntrain = int(0.8*Ns)
Ntest = int(0.2*Ns)
# Step 4: Split training and test datesets
xtrain = xdata[0:Ntrain,:].astype('float32')
ytrain = ydata[0:Ntrain,:].astype('float32')
xtest = xdata[Ns-Ntest:Ns,:].astype('float32')
ytest = ydata[Ns-Ntest:Ns,:].astype('float32')
model = load_model(r'C:\ITEEM\Submodel_WWT\model_AS.h5')
#model.summary()
## simulations from ANN
ypred = model.predict(xtest)
MSE = mean_squared_error(ytest,ypred)
R2_total = r2_score(ytest,ypred) 
MSE_raw = mean_squared_error(ytest,ypred,multioutput='raw_values')
R2_raw = r2_score(ytest,ypred, multioutput='raw_values')


'''plot figures'''
# fig = plt.figure(figsize=(4,3))
x = np.linspace(0, 1, 50)
ax1.plot(x,x,'--', color='black', label='_nolegend_')
ax1.text(0,1.1,'(a) AS',fontdict={'family': 'arial', 'size':10})
ax1.text(0,0.95,"Total R$^2$ = "+str(round(R2_total,3)),fontdict={'family': 'arial', 'size':10})
ax1.text(0,0.85,"MSE = "+str("{:.2e}".format(MSE)),fontdict={'family': 'arial', 'size':10})
ax1.scatter(ytest[:,0],ypred[:,0], s=16, color=color[0], marker="o")
# ax1.scatter(ytest[:,1],ypred[:,1], s=16, color=color[1], marker="x",alpha=1)
ax1.scatter(ytest[:,2],ypred[:,2], s=16, color=color[2], marker="D",alpha=1)
# ax1.scatter(ytest[:,3],ypred[:,3], s=16, color=color[3], marker="^",alpha=1)
# ax1.scatter(ytest[:,4],ypred[:,4], s=30, color=color[4], marker="+",alpha=1)
ax1.scatter(ytest[:,5],ypred[:,5], s=30, color=color[5], marker="*",alpha=1)
# ax1.set_xlabel('GPS-X simulation data (normalized value)',fontdict={'family': 'arial', 'size':10})
ax1.set_ylabel('ANN simulation\n(normalized value)', fontdict={'family': 'arial', 'size':10})

# ax2
data = pd.read_csv('C:\ITEEM\Submodel_WWT\simulation_data_csv\simulation_10000runs_ASCP_Dec.csv', skiprows=range(1,2))
data = data[(data.iloc[:,17]>data.iloc[:,2]*0.0052*0.99) & (data.iloc[:,17]<data.iloc[:,2]*0.0052*1.01)]
xdata = data.iloc[:,2:6]
ydata = data.iloc[:,9:17]
scalar = MinMaxScaler()
xdata = scalar.fit_transform(xdata).astype('float32')
ydata = scalar.fit_transform(ydata).astype('float32')
# Step 3: Define variables
Nop_var = len(xdata[0])
Noutput_var = len(ydata[0])
Ns = len(xdata[:])
Ntrain = int(0.8*Ns)
Ntest = int(0.2*Ns)
# Step 4: Split training and test datesets
xtrain = xdata[0:Ntrain,:]
ytrain = ydata[0:Ntrain,:].astype('float32')
xtest = xdata[Ns-Ntest:Ns,:]
ytest = ydata[Ns-Ntest:Ns,:].astype('float32')
model = load_model('C:\ITEEM\Submodel_WWT\model_ASCP.h5')
ypred = model.predict(xtest)
MSE = mean_squared_error(ytest,ypred)
R2_total = r2_score(ytest,ypred)
MSE_raw = mean_squared_error(ytest,ypred,multioutput='raw_values')
R2_raw = r2_score(ytest,ypred, multioutput='raw_values')

#A = history.history['loss']
'''plot figures'''
# fig = plt.figure(figsize=(4,3))
x = np.linspace(0, 1, 50)
ax2.plot(x,x,'--', color='black', label='_nolegend_')
# x = np.linspace(0, 1, 50)
ax2.plot(x,x,'--',color='black', label='_nolegend_')
ax2.text(0,1.1,'(b) ASCP',fontdict={'family': 'arial', 'size':10})
ax2.text(0,0.95,"Total R$^2$ = "+str(round(R2_total,3)),fontdict={'family': 'arial', 'size':10})
ax2.text(0,0.85,"MSE = "+str("{:.2e}".format(MSE)),fontdict={'family': 'arial', 'size':10})
ax2.scatter(ytest[:,0],ypred[:,0], s=16, color=color[0], marker="o")
# ax2.scatter(ytest[:,1],ypred[:,1], s=16, color=color[1], marker="x",alpha=1)
ax2.scatter(ytest[:,2],ypred[:,2], s=16, color=color[2], marker="D",alpha=1)
# ax2.scatter(ytest[:,3],ypred[:,3], s=16, color=color[3], marker="^",alpha=1)
# ax2.scatter(ytest[:,4],ypred[:,4], s=30, color=color[4], marker="+",alpha=1)
ax2.scatter(ytest[:,5],ypred[:,5], s=30, color=color[5], marker="*",alpha=1)
# ax2.set_xlabel('GPS-X simulation data (normalized value)',fontdict={'family': 'arial', 'size':10})
# ax2.set_ylabel('ANN simulation (normalized value)', fontdict={'family': 'arial', 'size':10})

# # ax4
# data = pd.read_csv('C:\ITEEM\Submodel_WWT\simulation_data_csv\simulation_10000runs_EBPR_basic_Oct.csv', skiprows=range(1,2))
# data = data[data.iloc[:,21]>1]
# # data = data[data.iloc[:,11]<20]
# data = data[(data.iloc[:,17]>data.iloc[:,2]*0.0052*0.99) & (data.iloc[:,17]<data.iloc[:,2]*0.0052*1.01)]
# xdata = data.iloc[:,2:6]
# ydata = data.iloc[:,9:17]
# scalar = MinMaxScaler()
# xdata = scalar.fit_transform(xdata).astype('float32')
# ydata = scalar.fit_transform(ydata).astype('float32')
# Nop_var = len(xdata[0])
# Noutput_var = len(ydata[0])
# Ns = len(xdata[:])
# Ntrain = int(0.8*Ns)
# Ntest = int(0.2*Ns)
# xtrain = xdata[0:Ntrain,:]
# ytrain = ydata[0:Ntrain,:].astype('float32')
# xtest = xdata[Ns-Ntest:Ns,:]
# ytest = ydata[Ns-Ntest:Ns,:].astype('float32')
# model = load_model('C:\ITEEM\Submodel_WWT\EBPR\model_EBPR_Dec.h5')
# ## simulations from ANN
# ypred = model.predict(xtest)
# MSE = mean_squared_error(ytest,ypred)
# R2_total = r2_score(ytest,ypred)
# #A = history.history['loss']
# # fig = plt.figure(figsize=(4,3))
# x = np.linspace(0, 1, 50)
# ax4.plot(x,x,'--',color='black', label='_nolegend_')
# ax4.text(0,1.1,'(c) EBPR_basic',fontdict={'family': 'arial', 'size':10})
# ax4.text(0,0.95,"Total R$^2$ = "+str(round(R2_total,3)),fontdict={'family': 'arial', 'size':10})
# ax4.text(0,0.85,"MSE = "+str("{:.2e}".format(MSE)),fontdict={'family': 'arial', 'size':10})
# ax4.scatter(ytest[:,0],ypred[:,0], s=16, color=color[0], marker="o")
# # ax4.scatter(ytest[:,1],ypred[:,1], s=16, color=color[1], marker="x",alpha=1)
# ax4.scatter(ytest[:,2],ypred[:,2], s=16, color=color[2], marker="D",alpha=1)
# # ax4.scatter(ytest[:,3],ypred[:,3], s=16, color=color[3], marker="^",alpha=1)
# # ax4.scatter(ytest[:,4],ypred[:,4], s=30, color=color[4], marker="+",alpha=1)
# ax4.scatter(ytest[:,5],ypred[:,5], s=30, color=color[5], marker="*",alpha=1)
# ax4.set_xlabel('GPS-X simulation data\n(normalized value)',fontdict={'family': 'arial', 'size':10})
# ax4.set_ylabel('ANN simulation\n(normalized value)', fontdict={'family': 'arial', 'size':10})

# ax5
data = pd.read_csv('C:\ITEEM\Submodel_WWT\simulation_data_csv\simulation_10000runs_EBPR_acetate_Oct.csv', skiprows=range(1,2))
# q1, q3 = np.percentile(data.iloc[:,11],[25,75])
# iqr = q3 - q1
# lower_bound = q1 -(1.5 * iqr) 
# upper_bound = q3 +(1.5 * iqr) 
data = data[data.iloc[:,21]>1]
data = data[data.iloc[:,11]<20]
data = data[data.iloc[:,9]>0.001]
data = data[(data.iloc[:,17]>data.iloc[:,2]*0.0052*0.99) & (data.iloc[:,17]<data.iloc[:,2]*0.0052*1.01)]
xdata = data.iloc[:,2:6]
ydata = data.iloc[:,9:17]
scalar = MinMaxScaler()
xdata = scalar.fit_transform(xdata).astype('float32')
ydata = scalar.fit_transform(ydata).astype('float32')
Nop_var = len(xdata[0])
Noutput_var = len(ydata[0])
Ns = len(xdata[:])
Ntrain = int(0.8*Ns)
Ntest = int(0.2*Ns)
xtrain = xdata[0:Ntrain,:]
ytrain = ydata[0:Ntrain,:].astype('float32')
xtest = xdata[Ns-Ntest:Ns,:]
ytest = ydata[Ns-Ntest:Ns,:].astype('float32')
model = load_model('C:\ITEEM\Submodel_WWT\EBPR_acetate\model_EBPR_acetate_Dec_exlowN.h5')
## simulations from ANN
ypred = model.predict(xtest)
MSE = mean_squared_error(ytest,ypred)
R2_total = r2_score(ytest,ypred)
MSE_raw = mean_squared_error(ytest,ypred,multioutput='raw_values')
R2_raw = r2_score(ytest,ypred, multioutput='raw_values')

#A = history.history['loss']
# fig = plt.figure(figsize=(4,3))
x = np.linspace(0, 1, 50)
ax5.plot(x,x,'--',color='black', label='_nolegend_')
ax5.text(0,1.1,'(d) EBPR_acetate',fontdict={'family': 'arial', 'size':10})
ax5.text(0,0.95,"Total R$^2$ = "+str(round(R2_total,3)),fontdict={'family': 'arial', 'size':10})
ax5.text(0,0.85,"MSE = "+str("{:.2e}".format(MSE)),fontdict={'family': 'arial', 'size':10})
ax5.scatter(ytest[:,0],ypred[:,0], s=16, color=color[0], marker="o")
# ax5.scatter(ytest[:,1],ypred[:,1], s=16, color=color[1], marker="x",alpha=1)
ax5.scatter(ytest[:,2],ypred[:,2], s=16, color=color[2], marker="D",alpha=1)
# ax5.scatter(ytest[:,3],ypred[:,3], s=16, color=color[3], marker="^",alpha=1)
# ax5.scatter(ytest[:,4],ypred[:,4], s=30, color=color[4], marker="+",alpha=1)
ax5.scatter(ytest[:,5],ypred[:,5], s=30, color=color[5], marker="*",alpha=1)
ax5.set_xlabel('GPS-X simulation data\n(normalized value)',fontdict={'family': 'arial', 'size':10})
ax5.set_ylabel('ANN simulation\n(normalized value)', fontdict={'family': 'arial', 'size':10})

#ax6
# Step 1: Load data
data = pd.read_csv('C:\ITEEM\Submodel_WWT\simulation_data_csv\simulation_10000runs_EBPR_StR_Oct.csv', skiprows=range(1,2))
# data = pd.read_csv('C:\ITEEM\Submodel_WWT\simulation_data_csv\simulation_10000runs_EBPR_FBR_Oct.csv', skiprows=range(1,2))

# remove outliers for xmghpo4RL2 < 1 gMgHPO4.3H2O/m3
data = data[data.iloc[:,21]>1]
data = data[data.iloc[:,11]!=1.38062]
data = data[data.iloc[:,13]!=9.69921]
data = data[data.iloc[:,11]<15]

xdata = data.iloc[:,2:6]
ydata = data.iloc[:,9:18]
# Step 2: Normalize data
scalar = MinMaxScaler()
xdata = scalar.fit_transform(xdata).astype('float32')
ydata = scalar.fit_transform(ydata).astype('float32')
# Step 3: Define variables
Nop_var = len(xdata[0])
Noutput_var = len(ydata[0])
Ns = len(xdata[:])
Ntrain = int(0.8*Ns)
Ntest = int(0.2*Ns)
# Step 4: Split training and test datesets
xtrain = xdata[0:Ntrain,:]
ytrain = ydata[0:Ntrain,:].astype('float32')
xtest = xdata[Ns-Ntest:Ns,:]
ytest = ydata[Ns-Ntest:Ns,:].astype('float32')
model = load_model('C:\ITEEM\Submodel_WWT\model_EBPR_StR.h5')
#model.summary()
## simulations from ANN
ypred = model.predict(xtest)
MSE = mean_squared_error(ytest,ypred)
R2_total = r2_score(ytest,ypred)
R2_raw = r2_score(ytest,ypred, multioutput='raw_values')
MSE_raw = mean_squared_error(ytest,ypred, multioutput='raw_values')
MSE_raw[0], MSE_raw[2], MSE_raw[5], MSE_raw[6]
R2_raw[0], R2_raw[2], R2_raw[5], R2_raw[6]

#A = history.history['loss']
# fig = plt.figure(figsize=(4,3))
x = np.linspace(0, 1, 50)
ax6.plot(x,x,'--', color='black', label='_nolegend_')
ax6.text(0,1.1,'(e) EBPR_StR',fontdict={'family': 'arial', 'size':10})
ax6.text(0,0.95,"Total R$^2$ = "+str(round(R2_total,3)),fontdict={'family': 'arial', 'size':10})
ax6.text(0,0.85,"MSE = "+str("{:.2e}".format(MSE)),fontdict={'family': 'arial', 'size':10})
ax6.scatter(ytest[:,0],ypred[:,0], s=16, color=color[0], marker="o", label='Nitrate')
# ax6.scatter(ytest[:,1],ypred[:,1], s=16, color=color[1], marker="x", label='TSS')
ax6.scatter(ytest[:,2],ypred[:,2], s=16, color=color[2], marker="D",alpha=1, label='TP')
# ax6.scatter(ytest[:,3],ypred[:,3], s=16, color=color[3], marker="^",alpha=1, label='COD')
# ax6.scatter(ytest[:,4],ypred[:,4], s=30, color=color[4], marker="+",alpha=1, label='TN')
ax6.scatter(ytest[:,5],ypred[:,5], s=30, color=color[5], marker="*",alpha=1, label='Hauled sludge')
ax6.scatter(ytest[:,6],ypred[:,6], s=30, color=color[6], marker="s",alpha=1, label='Struvite\n(StR only)')
ax6.set_xlabel('GPS-X simulation data\n(normalized value)',fontdict={'family': 'arial', 'size':10})

# ax3.axis('off')
# ax4.axis('off')
# leg = ax4.legend()
handles, labels = ax6.get_legend_handles_labels()
# labels = ['Nitrate','TSS', 'TP','COD','TN','Hauled sludge', 'Struvite\n(StR only)']
# overall legend
fig.legend(handles, labels, bbox_to_anchor=(0.4, .72, 0.5, 0.4),
                  title='Simulation outputs', fontsize=10, ncol=4)
fig.tight_layout()
# leg.get_frame().set_edgecolor('black')

fig.savefig(r'C:\ITEEM\Submodel_WWT\SDD_analysis\figures\SDD_Dec2020\Fig3_N_P_only.tif', dpi=300, bbox_inches = 'tight')
plt.show()

