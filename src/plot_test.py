from keras.layers import RNN, Activation,LSTM, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

##########################################################################################

c = 1
color2 = ['b','#D81B60', '#FF9C07']
color = ['g','#1E88E5','#004D40']
labels = ['', 'Actual PV', 'Actual Consumption']
labels2 = ['', 'Predicted PV', 'Predicted Consumption']
lstyle = [ ':', '-', '-']
lstyle2 = ['','--','--']

##########################################################################################
#Parameters
name = 'RMSE for this model :'
name2 = 'MAPE for this model :'
n_lag = 48
n_seq = 1
N_LAG = n_lag
N_SEQ = n_seq
filename ='total_austin.csv'
verbose, epochs, batch_size = 2, 50 , 16
###########################################################################################
#Load Data
#Code adapted from machinelearningmastery tutorials
###########################################################################################
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
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
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = math.sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.3f' % s for s in scores])
	print('%s: [%.4f] %s' % (name, score, s_scores))

def evaluate_forecasts1(actual, predicted):
	#scores = list()
	#for i in range(actual.shape[1]):
		#if(actual[:, i]!=0 and actual[:, i] is not None):
		# calculate mse
		#	mape = 100*(actual[:, i] - predicted[:, i])/actual[:, i]
		# calculate rmse
		# store
		#scores.append(mape)
	# calculate overall RMSE
	s = 0
	total1 = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			if((actual[row, col] > 0) and (actual[row, col] is not None)):
				s += abs(actual[row, col] - predicted[row, col])/actual[row, col] 
				
				total1 += 1
	print(s)
	print(total1)			
	score = 100 * s / total1
	return score
# summarize scores
def summarize_scores1(name2, score):
	#s_scores = ', '.join(['%.3f' % s for s in scores])
	print('%s: %.4f' % (name2, score))


  
#======================================================================================================================
test_hours_to_plot = 192
print('Plotting predictions...')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#======================================================================================================================
# load the dataset
for i in range(1,3):
	c = i
	if c==1:
		case = 'solar'
		L = ['local_15min','solar']
	elif c==2:
		case = 'grid'
		L = ['local_15min','grid']
	else:
		case = 'car'

	modelname = 'models/centralizedModel_'+case+'.h5'

	dataframe = pd.read_csv(filename, usecols=L, engine='python',infer_datetime_format=True, parse_dates=['local_15min'], index_col=['local_15min'])
	print(dataframe.head())
	dataset = dataframe.values

#Split it 
	train_size = int(len(dataset) * 0.9)
	test_size = len(dataset) - train_size
	traind, testd = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	traind = scaler.fit_transform(traind)
	testd = scaler.transform(testd)

#Turn the data into the supervised learning shape  

	train,test = series_to_supervised(traind, n_lag, n_seq), series_to_supervised(testd, n_lag, n_seq)
	train_values = train.values
	test_values = test.values

# split into inputs and steps to predict
	trainX,trainY= train_values[:, 0:n_lag], train_values[:, n_lag:]
	testX,testY= test_values[:, 0:n_lag], test_values[:, n_lag:]
# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
	testX = numpy.reshape(testX, (testX.shape[0],testX.shape[1],1))

	model =  keras.models.load_model(modelname)

	predicted = model.predict(testX)
	predicted = scaler.inverse_transform(predicted)
	testY = scaler.inverse_transform(testY)
	#score, scores = evaluate_forecasts(predicted,testY)
	#summarize_scores(name,score,scores)
	score2 = evaluate_forecasts1(predicted,testY)
	summarize_scores1(name2,score2)

	predictions = numpy.array(predicted)
	actual = [row for row in testY]	
	actual2=numpy.array(actual)
	ax.plot(actual2[:test_hours_to_plot,0], color = color[i] , label=labels[i], lw=2, linestyle = lstyle[i])  # plot actual test series

	plt.plot(predictions[:test_hours_to_plot , 0], color = color2[i] , lw=2, label=labels2[i], linestyle = lstyle2[i])

plt.grid()
#, fontsize='large'
plt.legend(loc=4,fontsize = 'large')
plt.ylabel('Power',fontsize = 'x-large')
plt.xlabel('Time Steps',fontsize = 'x-large')
#plt.title('Predictions for first {0} steps in test set'.format(test_hours_to_plot ))
plt.show()

