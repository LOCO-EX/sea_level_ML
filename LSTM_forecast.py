#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#
"""
Created on Mon Sep 27 11:07:21 2021

I use the example given in
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

To do a forecast of the tidally-averaged sea level as a function of wind speed and direction, and tidal information.


@author: matias
"""

#%% prepare data for lstm

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# %% convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, n_f=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in-1, 0, -1):
		cols.append(df.loc[:,0:(n_f-1)].shift(i))
    #
	for i in range(n_in, 0, -1):    
		names += [('var%d(t-%d)' % (j+1, i-1)) for j in range(n_f)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % data.shape[1])]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_out)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# %%
# load dataset
dataset = read_csv('./data/tidal_averages.csv', header=0, index_col=0)
#dataset = read_csv('../data_post_proc/tidal_averages.csv', header=0, index_col=0) #sea level, wind speed, sin(ang), cos(ang)
#order dataset to put target(sea level) in the last column
cols = list(dataset.columns.values); cols=cols[1:]+[cols[0]]; dataset=dataset[cols]
values = dataset.values #(14113,4)
values[0,3]=0

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
n_steps_in = 3  #specify the number of the previous time steps to use for the prediction = 1 in this case
n_steps_out = 1 #specify the number of time steps to predict = 1 in this case because we are predicting only 1 time step
n_features = 4 #number of features (variables) used to predict

# frame as supervised learning
reframed = series_to_supervised(scaled, n_steps_in, n_steps_out, n_features)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head()) #(nsamples, 4*(n_steps_in+n_steps_out))
reframed.shape

# %%
# split into train and test sets
nsamples=reframed.shape[0] #=14107
values = reframed.values
n_train_periods = int(nsamples*0.5) #70% for training
train = values[:n_train_periods, :]
test = values[n_train_periods:, :]
# split into input and outputs (works only with n_steps_in=n_steps_out=1)
n_obs = n_steps_in * n_features #(features=predictors) #1*3=3
#
#for predicting sea level at time t using predictors at time t---
#train_X, train_y = train[:, :n_obs], train[:, -n_features]
#test_X, test_y = test[:, :n_obs], test[:, -n_features]
#
#for predicting sea level at time t+1 using predictors at time t---
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
#
print(train_X.shape, train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_steps_in, n_features))
test_X = test_X.reshape((test_X.shape[0], n_steps_in, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#%%
# design network
model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
#model.add(LSTM(2, input_shape=(train_X.shape[1], train_X.shape[2]))) #=(n_steps_in,n_features)
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam') #mean absolute error "mse" "mae"
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# %% Make a prediction
yhat = model.predict(test_X)
test_X0 = test_X.reshape((test_X.shape[0], n_steps_in*n_features))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
inv_yhat = concatenate((test_X0,yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat[:,-(n_features+1):])
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y0 = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = concatenate((test_X0,test_y0), axis=1)
inv_y = scaler.inverse_transform(inv_y[:,-(n_features+1):])
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print('Test std: %.3f' % inv_y.std())


pyplot.plot(inv_y, inv_yhat,'o')
pyplot.xlabel("data")
pyplot.ylabel("prediction")
pyplot.grid()
pyplot.axis([500,900,500,900])
pyplot.axis("equal")
pyplot.show()


pyplot.plot(inv_y,'r',label="data")
pyplot.plot(inv_yhat,'b',label="prediction")
pyplot.legend()
pyplot.show()
