#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 06:36:13 2021

This script computes the tidal averages of sea level, harmonic reconstruction, sea level residual, wind speed, and wind direction

Details about the averaging being carried out correctly must still be checked (integration boundaries). 
Later, tidal averaging of fresh water discharge should be added.

@author: mduranmatute
"""



import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# %% Load data for sea level
SL = pd.read_csv('data/SL_DH_decomposed.csv')

t = np.array([time.mktime(tmp.timetuple()) for tmp in SL['time'][:].astype('datetime64')])
t = t[:]-t[0]

level = SL['sea_level_0m'][:] # Sea level
hrec  = SL['harmonic_rec'][:] # Harmonic reconstruction
res   = SL['residual'][:]     # Sea level residual (i.e. sea level minus harmonic reconstruction)  

# %% Define new time to integrate over tidal periods

TM2 = (12.42060120)*60*60

dT = 0.0125  #Close to 10 minutes, but fits better... 

periods = np.arange(0,14114,dT)+1 #14115 is the number of tidal periods in the time series

t_i = TM2*periods

# %% Interpolate to new time
level_i = np.interp(t_i,t,level)
hrec_i  = np.interp(t_i,t,hrec)
res_i   = np.interp(t_i,t,res)



# %% Make averaging

# First determine the limints of averaging given by a full period of M2 tide

def isInt(num):
    return np.equal(np.mod(num, 1), 0)

int_p = np.where(isInt(periods))[0] # int_p gives the index marking full tidal periods

# Now do the averaging 

level_av = [np.trapz(level_i[int_p[i]:int_p[i+1]+1],dx=dT) for i in np.arange(len(int_p)-1)]
hrec_av  = [np.trapz(hrec_i[int_p[i]:int_p[i+1]+1],dx=dT) for i in np.arange(len(int_p)-1)]
res_av   = [np.trapz(res_i[int_p[i]:int_p[i+1]+1],dx=dT) for i in np.arange(len(int_p)-1)]

# %%Plot some visual checks

# t_lim = 268272.0

# plt.plot(t[t<t_lim],level[t<t_lim],'r')
# plt.plot(t_i[t_i<t_lim],level_i[t_i<t_lim],'--g')
# plt.plot(t_i[int_p[0:7]],level_i[int_p[0:7]],'x')
# plt.plot((t_i[int_p[0:7]]+t_i[int_p[1:8]])/2,level_av[0:7],'o-')
# plt.show()

# plt.plot(t[t<t_lim],res[t<t_lim],'r')
# plt.plot(t_i[t_i<t_lim],res_i[t_i<t_lim],'--g')
# plt.plot(t_i[int_p[0:7]],res_i[int_p[0:7]],'x')
# plt.plot((t_i[int_p[0:7]]+t_i[int_p[1:8]])/2,res_av[0:7],'o-')
# plt.show()

# plt.plot(t[t<t_lim],hrec[t<t_lim],'r')
# plt.plot(t_i[t_i<t_lim],hrec_i[t_i<t_lim],'--g')
# plt.plot(t_i[int_p[0:7]],hrec_i[int_p[0:7]],'x')
# plt.plot((t_i[int_p[0:7]]+t_i[int_p[1:8]])/2,hrec_av[0:7],'o-')
# plt.show()

# %%

W = pd.read_csv('data/raw_data/Wind_data.csv')

gdr = np.pi/180 # useful to transform from degrees to radians

W['t'] = (W['Datenum'][:]-W['Datenum'][0])*24*60*60 #Define time in seconds

# %% Wind averaging
# Load the wind direction and perform some useful definitions
a = np.array(W['WindDirection'])  
a[a == 990] =0;

a_x = np.sin(np.pi/180*a)
a_y = np.cos(np.pi/180*a)
a_x[a == 990] = 0;
a_y[a == 990] = 0;

#predifine the output variables
mean_a_x = np.zeros([len(int_p)-1,])
mean_a_y = np.zeros([len(int_p)-1,])
trap_v   = np.zeros([len(int_p)-1,])

# This can be use in case you want to perform the wind averages with an offset 
# change the first number to the number of hours

t_off = 0*60*60; 

for i in np.arange(1,len(int_p)-1):

    # Find the indices for the averaging
    ind1W = np.where(W['t']>(t_i[int_p[i]]-t_off))[0][0]
    ind2W = np.where(W['t']<(t_i[int_p[i+1]]-t_off))[0][-1]
    
    NormfWD = ind2W - ind1W + 1 # The number of 
    
    
    mean_a_x[i] = np.nanmean(a_x[ind1W:ind2W+1])
    mean_a_y[i] = np.nanmean(a_y[ind1W:ind2W+1])

    trap_v[i] = np.array([np.trapz(W['WindSpeed_m_s_'][ind1W:ind2W+1])/NormfWD]) #average wind speed
    

a_mean = -180/np.pi*np.arctan2(mean_a_x,mean_a_y)
R = np.abs(mean_a_x**2 + mean_a_y**2);
std_angle = np.sqrt(-np.log(R))*180/np.pi;

# %% Split wind energy into sectors
a_sec = np.arange(22.5,360,45)

mws = np.zeros([a.size,8])

for i in np.arange(0,7):
	ind = (a>=a_sec[i]) & (a<a_sec[i+1])
	mws[ind,i+1]=W['WindSpeed_m_s_'][ind]
	
ind1 = (a<=a_sec[0]) 
ind2 = (a>a_sec[-1])
ind = np.logical_or(ind1,ind2)
mws[ind,0]=W['WindSpeed_m_s_'][ind]

MWS = np.zeros([len(int_p)-1,8])

for i in np.arange(1,len(int_p)-1):

    # Find the indices for the averaging
	ind1W = np.where(W['t']>(t_i[int_p[i]]-t_off))[0][0]
	ind2W = np.where(W['t']<(t_i[int_p[i+1]]-t_off))[0][-1]
	NormfWD = ind2W - ind1W + 1

	MWS[i,:] = np.array([np.trapz(mws[ind1W:ind2W+1,:]**3,axis = 0)/NormfWD]) #average wind speed



# %% Save data
d = {'sea_level_0m': level_av, 'wind_speed': trap_v, 'sine_wind_angle': mean_a_y, 'cosine_wind_angle': mean_a_x, 'std_wind_angle': std_angle}
df = pd.DataFrame(data=d)

df.to_csv('data/tidal_averages.csv')

d2 = {'sea_level_0m': level_av, 'E1': MWS[:,0], 'E2': MWS[:,1], 'E3': MWS[:,2], 'E4': MWS[:,3], 'E5': MWS[:,4], 'E6': MWS[:,5], 'E7': MWS[:,6], 'E8': MWS[:,7]}
df2 = pd.DataFrame(data=d2)

df2.to_csv('data/tidal_averages_ES.csv')


# %% Plot results
plt.scatter(a_mean,np.array(level_av),c=trap_v)
plt.show()

plt.scatter(a_mean,hrec_av,c=trap_v)
plt.show()

plt.scatter(a_mean,res_av,c=trap_v)
plt.show()

# %% Plot correlations

from scipy.stats import gaussian_kde

mask = np.where((a_mean<90) & (a_mean>45) & (std_angle<12) )



tmp1 = np.array(res_av)
tmp2 = np.array(level_av)

tmp = tmp2

np.corrcoef(trap_v[mask]**2/12,(tmp1[mask]))
np.corrcoef(trap_v[mask]**2/12,(tmp2[mask]))



trap_v[np.where(np.isnan(trap_v))]=0

xy = np.vstack([trap_v[mask],tmp1[mask]])
z1 = gaussian_kde(xy)(xy)

xy = np.vstack([trap_v[mask],tmp2[mask]])
z2 = gaussian_kde(xy)(xy)

plt.scatter(trap_v[mask]**2/12,(tmp1[mask]),c=z1)
plt.show()
plt.scatter(trap_v[mask]**2/12,(tmp2[mask]),c=z2)
plt.show()