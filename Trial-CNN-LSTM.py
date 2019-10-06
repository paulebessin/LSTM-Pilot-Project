# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:30:01 2019

@author: Paule Carelle
"""
# I- Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot 
import pandas as pd
from pandas import DataFrame
from pandas import concat
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import mean
from numpy import std
from numpy import concatenate
from sklearn.metrics import confusion_matrix


'''# Importing the dataset
Dataset = pd.read_csv('crew1_leftSeat_DA.csv')
Dataset['time'] = Dataset['time'].astype('timedelta64[s]')
Dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
Dataset.info()
Dataset.head()
Dataset.to_csv('crew1_DA1_new.csv')
 
#Data selection
train = pd.read_csv('crew1_DA1_new.csv', index_col='time')
train.drop(['Unnamed: 0'], axis=1, inplace=True)
train.index.name = 'time'

#Sample of the data
Sample = train.sample(frac=0.2,  replace=True, random_state=0)
Sample.sort_index(inplace=True)

# Encoding categorical data
values = Sample.values
labelencoder_1 = LabelEncoder()
values[:, 1] = labelencoder_1.fit_transform(values[:, 1])
labelencoder_2 = LabelEncoder()
values[:, 26] = labelencoder_2.fit_transform(values[:, 26])
values = values.astype('float32')

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
values[:,3:26] = scaler.fit_transform(values[:,3:26])

# Drop columns we don't want to predict
Final= pd.DataFrame(values)
Final.drop([0], axis=1, inplace=True)
Final.to_csv('Final.csv')'''

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# Importing the dataset
train = pd.read_csv('Final.csv')
train.drop(['Unnamed: 0'], axis=1, inplace=True)

# specify the number of lag seconds - equivalant to 35 seconds
n_timesteps = 50*5
n_features = 26

# frame as supervised learning
reframed = series_to_supervised(train, n_timesteps, 1)
print(reframed.shape)
 
# split into train and test sets
dataset = reframed.values

# Splitting the dataset into the Training set and Test set
train, test = train_test_split(dataset, test_size = 0.3, random_state = 0)

# split into input and outputs
n_obs = n_timesteps * n_features
X_train1, y_train = train[:, :n_obs], train[:, -n_features]
X_test1, y_test = test[:, :n_obs], test[:, -n_features]
print(X_train1.shape, len(X_train1), y_train.shape)

# fit and build and evaluate the combined CNN and LSTM model
#CNN Model for feature extraction and the LSTM Model for interpreting the features across time steps
def evaluate_classifier(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 50, 150
    n_timesteps, n_features, n_outputs = n_timesteps, X_train.shape[2], y_train.shape[0]
    # reshape data into time steps of sub-sequences
n_steps, n_length = 5, 50
X_train = X_train1.reshape((X_train1.shape[0], n_steps, n_length, n_features))
X_test = X_test1.reshape((X_test1.shape[0], n_steps, n_length, n_features))
    # Initialising the model
classifier = Sequential()
    #Adding the first layer of CNN
classifier.add(TimeDistributed(Conv1D(filters=45, kernel_size=3, activation='relu'), 
                               input_shape=(None,n_length,n_features)))
	#Adding the second layer of CNN and some Dropout regularisation
classifier.add(TimeDistributed(Conv1D(filters=45, kernel_size=3, activation='relu')))
classifier.add(TimeDistributed(Dropout(0.5)))
	#Adding a layer of MaxPooling and Flatten regularisation
classifier.add(TimeDistributed(MaxPooling1D(pool_size=2)))
classifier.add(TimeDistributed(Flatten()))
    # Adding the first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.5))
# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.5))
    # Adding a third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.5))
# Adding a fourth LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100))
classifier.add(Dropout(0.5))
    # Adding the output layer
classifier.add(Dense(1))
    # Compiling the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the model to the Training set
classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
               validation_data=(X_test, y_test))
    # Final evaluation of the model
_, accuracy = classifier.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
return("Accuracy: %.2f%%" % (accuracy[1]*100))



# Copy model and save
verbose, epochs, batch_size = 0, 50, 150
    n_timesteps, n_features, n_outputs = n_timesteps, X_train.shape[2], y_train.shape[1]
# reshape data into time steps of sub-sequences
n_steps, n_length = n_steps, n_length
X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))

# Initialising the model
classifier = Sequential()
#Adding the first layer of CNN
classifier.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), 
                               input_shape=(None,n_length,n_features)))
#Adding the second layer of CNN and some Dropout regularisation
classifier.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
classifier.add(TimeDistributed(Dropout(0.5)))
#Adding a layer of MaxPooling and Flatten regularisation
classifier.add(TimeDistributed(MaxPooling1D(pool_size=2)))
classifier.add(TimeDistributed(Flatten()))
# Adding the first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.5))
# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.5))
# Adding a third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.5))
# Adding a fourth LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100))
classifier.add(Dropout(0.5))
# Adding the output layer
classifier.add(Dense(n_outputs, activation='softmax'))
# Compiling the model
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the model to the Training set
model= classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
               validation_data=(X_test, y_test))
# save model to single file
classifier.save('cnn-lstm_model.h5')

# plot model
pyplot.plot(model.model['loss'], label='train')
pyplot.plot(model.model['val_loss'], label='test')
pyplot.legend()
pyplot.show()

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
X_test = X_test.reshape((X_test.shape[0], n_timesteps*n_features))
inv_ypred = concatenate((y_pred, X_test[:, -25:]), axis=1)
inv_ypred = scaler.inverse_transform(inv_ypred)
inv_ypred = inv_ypred[:,0]

# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, -25:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Making the Confusion Matrix
cm = confusion_matrix(inv_y, inv_ypred)
print(cm)

#Plot the confusion matrix
plot_confusion_matrix(cm, classes=["A", "B", "C", "D"],
                      title='Confusion matrix')

#Plot the prediction
pyplot.plot(inv_ypred)
pyplot.plot(inv_y)
pyplot.show()

#For a cleaner visualisation result, we can try the last 2 seconds = to 512 records
pyplot.plot(inv_yhat[-512:])
pyplot.plot(inv_y[-512:])
pyplot.show()


####### NEEDS A LITTLE BIT OF WORK BUT YOU CAN RUN IT AND SEE IF IT WORKS #########
#IV-Prediction on real Test Data

#load test data
test = pd.read_csv('test.csv', index_col='time')
test.drop(['Unnamed: 0'], axis=1, inplace=True)
test.index.name = 'time'

test_id = test['id']
test.drop(['id'], axis=1, inplace=True)

# Encoding categorical data
values_test = test.values
labelencoder_3 = LabelEncoder()
values_test[:, 1] = labelencoder_3.fit_transform(values_test[:, 1])
values_test = values_test.astype('float32')

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
values_test[:,3:26] = scaler.fit_transform(values[:,3:26])

# Drop columns we don't want to predict
Test= pd.DataFrame(values)
Test.drop([0], axis=1, inplace=True)

classifier1 = load_model('cnn-lstm_model.h5')

pred = classifier1.predict_proba(Test)
sub = pd.DataFrame(pred,columns=['A', 'B', 'C', 'D'])
sub['id'] = test_id
cols = sub.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub = sub[cols]
sub.to_csv("Test_prob.csv", index=False)