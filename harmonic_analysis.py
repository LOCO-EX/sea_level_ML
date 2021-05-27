# -*- coding: utf-8 -*-
"""
Script to perform the harmonic analysis of the sea level

This script read the sea level (corrected) raw data, performs harmonic analysis
and saves the tidal reconstruction and the residual in a new file



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
level = SL.CorrectedSeaLevel


# %% Perform harmonic analysis

#import netCDF4
import pytide
import numpy as np


#wt = pytide.WaveTable()
wt = pytide.WaveTable()

f, vu = wt.compute_nodal_modulations(time)
# You can also use a list of datetime.datetime objects
# wt.compute_nodal_modulations(list(time))

w = wt.harmonic_analysis(level, f, vu)

time_ms = [element.timestamp()+3600 for element in time]


hp = wt.tide_from_tide_series(time_ms, w)+np.mean(level)



# %%
import matplotlib.pyplot as plt
plt.plot(time[0:800],level[0:800])
plt.plot(time[0:800],hp[0:800])
plt.show()