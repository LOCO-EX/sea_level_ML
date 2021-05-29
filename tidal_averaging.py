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

dT = 0.0125

periods = np.arange(0,14115,0.0125)

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

