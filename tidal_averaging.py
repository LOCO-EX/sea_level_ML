#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 06:36:13 2021

@author: mduranmatute
"""



import pandas as pd
import time
import numpy as np

# %% Load data
SL = pd.read_csv('data/SL_DH_decomposed.csv')

t = np.array([time.mktime(tmp.timetuple()) for tmp in SL['time'][:].astype('datetime64')])
t = t[:]-t[0]

level = SL['sea_level_0m'][:]
hrec  = SL['harmonic_rec'][:]
res   = SL['residual'][:]

# %% Define new time

TM2 = (12.42060120)*60*60

dT = 0.0125  #Close to 10 minutes, but fits better... 

periods = np.arange(0,14115,dT) #14115 is the number of tidal periods in the time series

t_i = TM2*periods

# %% Interpolate to new time
level_i = np.interp(t_i,t,level)
hrec_i  = np.interp(t_i,t,hrec)
res_i   = np.interp(t_i,t,res)



# %% Make averaging

# First determine the limints of averaging given by a full period of M2 tide

def isInt(num):
    return np.equal(np.mod(num, 1), 0)

int_p = np.where(isInt(periods))[0]

level_av = [np.trapz(level_i[int_p[i]:int_p[i+1]],dx=dT) for i in np.arange(len(int_p)-1)]
hrec_av  = [np.trapz(hrec_i[int_p[i]:int_p[i+1]],dx=dT) for i in np.arange(len(int_p)-1)]
res_av   = [np.trapz(res_i[int_p[i]:int_p[i+1]],dx=dT) for i in np.arange(len(int_p)-1)]


# %%

W = pd.read_csv('data/raw_data/Wind_data.csv')

gdr = np.pi/180

W['t'] = (W['Datenum'][:]-W['Datenum'][0])*24*60*60 #Define time in seconds

a = np.array(W['WindDirection'])

a_x = np.sin(np.pi/180*a)

a_y = np.cos(np.pi/180*a)
a_x[a == 990] = np.nan;
a_y[a == 990] = np.nan;

mean_a_x = np.zeros([len(int_p),1])
mean_a_y = np.zeros([len(int_p),1])
trap_v   = np.zeros([len(int_p),1])

for i in np.arange(1,len(int_p)-1):

    ind1W = np.where(W['t']>t_i[int_p[i]])[0][0]
    ind2W = np.where(W['t']<t_i[int_p[i+1]])[0][-1]
    
    NormfWD = ind2W - ind1W;
    
    mean_a_x[i] = np.nanmean(a_x[ind1W:ind2W+1]);
    mean_a_y[i] = np.nanmean(a_y[ind1W:ind2W+1]);

    trap_v[i] = np.array([np.trapz(W['WindSpeed_m_s_'][ind1W:ind2W+1],dx=NormfWD)])
    

#W_EX_i = np.interp(t_i,W['t'],W['EnergyX_MJ_'])
#W_EY_i = np.interp(t_i,W['t'],W['EnergyX_MJ_'])

#W_Di_i = np.interp(t_i,W['t'],W['WindDirection'])
#W_Sp_i = np.interp(t_i,W['t'],W['WindSpeed_m_s_'])

#W_D_av = [np.arctan(np.sum(np.sin(gdr*W['WindDirection'][int_p[i]:int_p[i+1]]))/np.sum(np.cos(gdr*W['WindDirection'][int_p[i]:int_p[i+1]]))) for i in np.arange(len(int_p)-1)]

#W_Sp_av = np.array([np.trapz(W_Sp_i[int_p[i]:int_p[i+1]],dx=dT) for i in np.arange(len(int_p)-1)])

#W_EX_av = np.array([np.trapz(W_EX_i[int_p[i]:int_p[i+1]],dx=dT) for i in np.arange(len(int_p)-1)])

#a_mean = 180/np.pi*np.arctan(mean_a_x/mean_a_y)
a_mean = 180/np.pi*np.arctan2(mean_a_x,mean_a_y)
