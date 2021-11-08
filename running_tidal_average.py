#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:05:46 2021


This script computes a running tidal averages of sea level. 


@author: Matias Duran Matute
"""





import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# %% Load data for sea level
SL = pd.read_csv('data/SL_DH_decomposed.csv')

t = np.array([time.mktime(tmp.timetuple()) for tmp in SL['time'][:].astype('datetime64')])
t = t[:]-t[0]

level = SL['sea_level_0m'][:]

# %% Define averaging function
def running_mean(x, N):
    sum = np.sum(np.insert(x, 0, 0)) 
    return (sum[N:] - sum[:-N]) / float(N)



# %% Perform averaging
# Sea level data is every 10 minutes. To perform an average every 2 tidal periods we must average, we take 149 data points.

groups = [level[x:x+149] for x in range(0, len(level), 6)]

# Simple math to calculate the means
level_da = np.array([sum(group)/len(group) for group in groups])[0:-1]
t_da = t[6::6]


# %% Save data

d = {'level_da': level_da, 'time_da': t_da}
df = pd.DataFrame(data=d)

df.to_csv('data/running_daily_level.csv')