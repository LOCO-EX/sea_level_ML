# -*- coding: utf-8 -*-
"""
Script to perform the harmonic analysis of the sea level

This script reads the sea level (corrected) raw data to perform harmonic analysis
and filtering of the water level.

It saves the tidal reconstruction, the residual and the filtered sea level to 
a file. 



Must intall pytide
conda install pytide -c conda-forge


"""


import pandas as pd


#%% Function to convert MATLAB's datenum to Python's datetime
import datetime as dt


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return dt.datetime.fromordinal(int(datenum)) \
           + dt.timedelta(days=days) \
           - dt.timedelta(days=366)


# %% Load Data

SL = pd.read_csv('data/raw_data/SL_DH_data.csv')

time  = list(map(datenum_to_datetime,SL.datenum))
level = SL.CorrectedSeaLevel[:]-694.6

years=[element.year for element in time]



j2 = 473472

level[j2]=(level[j2-1]+level[j2+1])/2
# %% Perform harmonic analysis

#import netCDF4
import pytide
import numpy as np




wt = pytide.WaveTable()
#To test difference with t_tide these are the constituents with snr>10 for 2015
#wt = pytide.WaveTable(["O1", "P1", "K1", "Mu2", "N2", "Nu2", "M2", "L2", "S2", "K2", "MO3", "MN4", "M4", "MS4", "2MK6", "2MN6", "M6", "2MS6", "M8"])


hp = list()


for year in np.unique(years):

    ind = np.nonzero(np.array(years) == year )[0]

    time_tmp = [time[i] for i in ind]
    level_tmp = [level[i] for i in ind]

    f, vu = wt.compute_nodal_modulations(time_tmp)
    # You can also use a list of datetime.datetime objects
    # wt.compute_nodal_modulations(list(time))

    w = wt.harmonic_analysis(level_tmp, f, vu)

    time_ms = [element.timestamp()+3600 for element in time_tmp]

    hp = hp + list(wt.tide_from_tide_series(time_ms, w)+np.mean(level_tmp))

res = level - hp +  694.6    

d = {'time': time, 'sea_level_0m': level +  694.6 , 'harmonic_rec': hp, 'residual': res}
df = pd.DataFrame(data=d)

# %%
#import matplotlib.pyplot as plt
#plt.plot(time[0:800],level[0:800])
#plt.plot(time[0:800],hp[0:800])
#plt.show()

# %% filtering the tide
    
from oceans.filters import lanc

freq = 1./40/6  # Hours
window_size = 6*(96+1+96)
pad = np.zeros(window_size) * np.NaN

win = lanc(window_size, freq)
res = np.convolve(win, df['sea_level_0m']-694.6, mode='same')

df['low'] = res + 694.6
df['high'] = df['sea_level_0m'] - df['low']

df['pandas_l'] = df['sea_level_0m'].rolling(window = 240, center=True, min_periods=1).mean()
df['pandas_h'] = df['sea_level_0m'] - df['pandas_l']

# %% This part using iris uses too much memory

#import iris
#from iris.pandas import as_cube

#cube = as_cube(df['sea_level'])
#low = cube.rolling_window('index',
#                        iris.analysis.SUM,
#                        len(wt),
#                        weights=wt)

#df['iris_l'] = np.r_[pad, low.data, pad]
#df['iris_h'] = df['sea_level'] - df['iris_l']


# %%
df.to_csv('data/SL_DH_decomposed.csv')