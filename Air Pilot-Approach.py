# -*- coding: utf-8 -*
#!/usr/bin/env python3-
"""
Created on Tue Apr  2 19:49:29 2019

@author: Paule Carelle
"""
# I- Data Preprocessing

# Importing the libraries
#import matplotlib.pyplot as plt
#from matplotlib import pyplot
#from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from numpy import array
from numpy import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import mean
from numpy import std
from numpy import concatenate
from sklearn.metrics import confusion_matrix
 
#Data selection
train = pd.read_csv('Train_new.csv', index_col='time')
train.index.name = 'time'
train.sort_index(inplace=True)
train.drop(['Unnamed: 0','crew','experiment'], axis=1, inplace=True)

# Encoding categorical data
values = train.values
labelencoder = LabelEncoder()
values[:, 24] = labelencoder.fit_transform(values[:, 24])
values = values.astype('float32')

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
values[:, 0:24] = scaler.fit_transform(values[:, 0:24])

#Define X and Y
XFinal= values[:, 0:24]
Y= output = values[:, 24]

#Fix imbalance in event
XFinal, Y = SMOTE().fit_resample(XFinal, Y.ravel())

#Set inputs and outputs
inputs = XFinal.reshape((len(XFinal), 24))
output = Y.reshape((len(Y), 1))

# horizontally stack columns
dataset = hstack((inputs, output))

# split into train and test sets

n_train_minutes = 360*3*256 #train on first 18 minutes
train = dataset[:n_train_minutes, :]
test = dataset[n_train_minutes:, :]

# split into input and outputs
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# one hot encode y
X_train = to_categorical(X_train)
y_test = to_categorical(y_test)

# Initialising the model
classifier = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
verbose, epochs, batch_size = 0, 50, 250
n_steps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train
classifier.add(LSTM(units = 300, return_sequences = True, 
                    input_shape = (n_steps, n_features)))
classifier.add(Dropout(0.3))

# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 300, return_sequences = True))
classifier.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 300, return_sequences = True))
classifier.add(Dropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 300))
classifier.add(Dropout(0.3))

# Adding the output layer
classifier.add(Dense(units = 300, activation='relu'))
classifier.add(Dense(1, activation='softmax'))

# Compiling the model
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()
 

# Fitting the model to the Training set
history = classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
               verbose=verbose, validation_data=(X_test, y_test))

# Final evaluation of the model
accuracy = classifier.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print("Accuracy: %.2f%%" % (accuracy[1]*100))

# plot history for loss and accuracy of the model
#pd.DataFrame(history.history).plot(figsize=(8, 5))
#plt.grid(True) 
#plt.gca().set_ylim(0,1) 
#plt.show()

#2nd history plot visualization
# list all data in history
#print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

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
		score = accuracy(X_train, X_test, y_train, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# III- Making the predictions 

# Getting prediction for test set
y_pred = classifier.predict(X_test)
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

#Plot the prediction
#pyplot.plot(inv_ypred)
#pyplot.plot(inv_y)
#pyplot.show()

####### NEEDS A LITTLE BIT OF WORK BUT YOU CAN RUN IT AND SEE IF IT WORKS #########
#IV-Prediction on real Test Data

#load test data
test = pd.read_csv('test.csv', index_col='time')
test.drop(['Unnamed: 0'], axis=1, inplace=True)
test.index.name = 'time'

test_id = test['id']
test.drop(['id', 'crew', 'experiment'], axis=1, inplace=True)


# Feature Scaling
values_test = test.values
scaler = MinMaxScaler(feature_range=(0, 1))
values_test[:,0:2] = scaler.fit_transform(values[:,0:24])

#Predict probabilities of Ids in Test data
Test= pd.DataFrame(values)
pred = classifier.predict_proba(Test)


sub = pd.DataFrame(pred,columns=['A', 'B', 'C', 'D'])
sub['id'] = test_id
cols = sub.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub = sub[cols]
sub.to_csv("Test_prob.csv", index=False)
