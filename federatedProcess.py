import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow_federated import python as tff
import random
import collections

from tensorflow.keras.layers import RNN, Activation, LSTM, Dropout, Dense
from keras.models import Model
from tensorflow.keras.models import Sequential

tf.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()
np.random.seed(0)
tf.random.set_random_seed(0)

ratio = 0.9
N_LAG = 48
N_SEQ = 4

path='clients_data_austin/dataTrain.h5'
data = tff.simulation.HDF5ClientData(path)
#print(data.client_ids)

BATCH_SIZE = 16
BUFFER_SIZE = 100

NSubset = 3
NUMROUNDS = 25
#code can be adjusted to allow multiple runs with different total clients
NUMBER_CLIENTS = [6]
#change to model saving path
modelname = 'model.h5'
# Using a namedtuple with keys x and y as the output type of the
# dataset keeps both TFF and Keras happy:
BatchType = collections.namedtuple('BatchType', ['x', 'y'])


def accesser(x):
  return tf.cast(x['traindata'],tf.float32)

def split_input_target(chunk):
  #print(chunk)
  input_seq = tf.map_fn(lambda x: x[:N_LAG], chunk)
  inputX=tf.reshape(input_seq,[BATCH_SIZE,N_LAG,1])
  #print(input_seq)
  targetY = tf.map_fn(lambda x: x[N_LAG:], chunk)
  #targetY=tf.reshape(target_seq,[BATCH_SIZE,N_SEQ,1])
  #print(target_seq)
  return BatchType(inputX, targetY)


#addd .repeat(nbr_epochs) if training for more than 1 local epoch
def preprocess(dataset):
  #return(dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).map(split_input_target))
  #.repeat(2)
  return(dataset.map(accesser)
         .apply(tf.data.experimental.unbatch())
                       .batch(N_LAG+N_SEQ, drop_remainder=True)
                       .batch(BATCH_SIZE, drop_remainder=True)
                       .map(split_input_target))

#choose client id to give as an example so as the model knows what to predict
raw_example_dataset = data.create_tf_dataset_for_client('661')
example_dataset = preprocess(raw_example_dataset)
print(example_dataset.output_types, example_dataset.output_shapes)
print(example_dataset)


def model_fn():
  model = Sequential()
  model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(N_LAG, 1)))
  model.add(Dropout(0.2))
  #model.add(LSTM(200, activation='relu', return_sequences=True))
  #model.add(Dropout(0.2)) 
  model.add(LSTM(200, activation='relu'))
  model.add(Dropout(0.2)) 
  model.add(Dense(N_SEQ,activation='relu'))
  model.compile(loss='mse', optimizer='adam')
  print('----------------------Model Compilation Done---------------------')
  x=tf.constant(np.random.rand(BATCH_SIZE,N_LAG,1),dtype=tf.float32)
  y=tf.constant(np.random.rand(BATCH_SIZE,N_SEQ),dtype=tf.float32)
  sample_batch = collections.OrderedDict([('x', x), ('y', y)]) 
  r=tff.learning.from_compiled_keras_model(model, sample_batch)
  print('----------------------TFF model created--------------------------')
  return r

#----------------------------------------------------------------------------------------


#order the clients ids
L=data.client_ids
clientsI=[]
for i in range(len(L)):
  clientsI.append(int(L[i]))
clientsI.sort()
traindata = []
clients = [str(i) for i in clientsI]

MetricsSave = [[]]
counter = 0


for x in NUMBER_CLIENTS:
  print('---------Testing for nbr of clients : ', x)
  for i in range(x):
    print('creating data for client',clients[i])
    traindata.append(data.create_tf_dataset_for_client(clients[i]))
  
  trainproc = [preprocess(k) for k in traindata]
  print(trainproc)

  print('***************************Training***************************')
  trainer = tff.learning.build_federated_averaging_process(model_fn)
  state = trainer.initialize()
  for i in range(NUMROUNDS):
    clients_in_subset= random.sample(range(0, x), NSubset)
    print(clients_in_subset)
    data_in_subset = [ trainproc[k] for k in clients_in_subset]
    state, metrics = trainer.next(state, data_in_subset)
    print('round {:2d}, metrics={}'.format(i, metrics))
#    f.write('round %d loss %f\r\n' % (i,metrics))
    MetricsSave[counter].append(metrics)
  counter+=1  
print(MetricsSave)

#f.close()

#save model
keras_model = Sequential()
keras_model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(N_LAG, 1)))
keras_model.add(Dropout(0.2))
#keras_model.add(LSTM(200, activation='relu', return_sequences=True))
#keras_model.add(Dropout(0.2)) 
keras_model.add(LSTM(200, activation='relu'))
keras_model.add(Dropout(0.2)) 
keras_model.add(Dense(N_SEQ,activation='relu'))
keras_model.compile(loss='mse', optimizer='adam')
#tff.learning.assign_weights_to_keras_model(keras_model, state.model)
keras_model.set_weights(tff.learning.keras_weights_from_tff_weights(state.model))
keras_model.save(modelname)
