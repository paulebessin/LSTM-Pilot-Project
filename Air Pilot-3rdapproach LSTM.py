# -*- coding: utf-8 -*-
#!/usr/bin/env python3-
"""
Created on Fri Apr 12 12:17:41 2019

@author: Paule Carelle
"""
# I- Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from numpy import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import mean
from numpy import std
from numpy import concatenate
from sklearn.metrics import confusion_matrix

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# Importing the dataset
Dataset = pd.read_csv('Train_withoutoutliers.csv')
Dataset['time'] = Dataset['time'].astype('timedelta64[s]')
Dataset.info()
Dataset.head()
Dataset.to_csv('Train_new.csv')
 
#Data selection
train = pd.read_csv('Train_new.csv', index_col='time')
train.drop(['Unnamed: 0'], axis=1, inplace=True)
train.index.name = 'time'

#Sampling Train and Test Data
dataset = train.sample(frac=0.4, replace=True, random_state=0)

# Encoding categorical data
values = dataset.values
labelencoder_1 = LabelEncoder()
values[:, 1] = labelencoder_1.fit_transform(values[:, 1])
labelencoder_2 = LabelEncoder()
values[:, 26] = labelencoder_2.fit_transform(values[:, 26])
values = values.astype('float32')

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
values[:, 3:26] = scaler.fit_transform(values[:, 3:26])

# Drop columns we don't want to predict
Final= pd.DataFrame(values)
Final.drop([0], axis=1, inplace=True)

#Set inputs and outputs
inputs = Final.iloc[:, 1:26].values
output = Final.iloc[:, 25].values
inputs = inputs.reshape((len(inputs), 25))
output = output.reshape((len(output), 1))

# horizontally stack columns
dataset = hstack((inputs, output))

# Number of time steps- equivalant to 35 seconds
n_steps = 256*35

# convert into input/output
X, y = split_sequences(dataset, n_steps)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

# fit and evaluate a model
def evaluate_classifier(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 50, 150
    n_steps, n_features, n_outputs = n_steps, X_train.shape[2], y_train.shape[0]
    # Initialising the model
    classifier = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = 100, return_sequences = True, 
                    input_shape = (n_steps, n_features)))
    classifier.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = 100, return_sequences = True))
    classifier.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = 100, return_sequences = True))
    classifier.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = 100))
    classifier.add(Dropout(0.2))
    # Adding the output layer
    classifier.add(Dense(n_outputs, activation='softmax'))
    # Compiling the model
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Fitting the model to the Training set
    classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
               validation_data=(X_test, y_test))
    # Final evaluation of the model
    _, accuracy = classifier.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return("Accuracy: %.2f%%" % (accuracy[1]*100))

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_classifier(X_train, X_test, y_train, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# III- Making the predictions 

# Getting prediction for test set
y_pred = evaluate_classifier.predict(X_test)
print(y_pred)
# invert scaling for forecast
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
inv_ypred = concatenate((y_pred, X_test[:, 1:]), axis=1)
inv_ypred = scaler.inverse_transform(inv_ypred)
inv_ypred = inv_ypred[:,0]

# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
