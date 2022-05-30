from sklearn.preprocessing import MinMaxScaler
import h5py
import os
import pandas as pd
import joblib
from datetime import datetime
import pytz



path1='data_CA/'
path2 = 'data_1min_austin/'

c = 5

PATH = path2
PATH1 = path2
if c == 3:
    L = [2,3]
    case = 'solar'
    PATH = PATH +case+'/'
    print(PATH)
elif c == 4:
    L = [2,3,4]
    case = 'grid'
    PATH2 = PATH +'all'
    PATH = PATH + 'solar'

elif c ==5:
    L=[2,5]
    case = 'car'


outputTrain = 'clients_data_austin'+'/dataTrain_1min_'+case+'.h5'
outputTest = 'clients_data_austin'+'/dataTest_1min_'+case+'.h5'
counter = 0


ratio = 0.9
N_LAG = 30
N_SEQ = 15
BATCH_SIZE = 16
BUFFER_SIZE = 100
n_lag = N_LAG
n_seq = N_SEQ
utc = pytz.UTC

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
   
mindate = datetime.strptime('2018-05-20 23:55:31-06:00', '%Y-%m-%d %H:%M:%S%z')
#plswrk = pytz.timezone('US/Pacific')
#mindate = pytz.localize(mindate1)
maxdate = datetime.strptime('2018-06-01 00:00:00-06:00', '%Y-%m-%d %H:%M:%S%z')
#maxdate = pytz.utc.localize(maxdate1)
with h5py.File(outputTrain, 'w') as tr, h5py.File(outputTest, 'w') as ts:
  GTrain = tr.create_group('examples')
  GTest = ts.create_group('examples')
  filenameL1 = os.listdir(PATH)
  #filenameL2 = os.listdir(PATH2)
  #for filename1 in os.listdir(path1):
  for filename1 in filenameL1:
    print('==========================================',filename1)
    dataframe = pd.read_csv(os.path.join(PATH,filename1), usecols=L, engine='python',infer_datetime_format=True, parse_dates=['localminute'])
    #, index_col=['localminute']
    print(dataframe.head())
    df = dataframe[(dataframe['localminute']>mindate ) & (dataframe['localminute']<maxdate)]
    print(df.head())
    print(df['localminute'].max())
    print(df['localminute'].min())
    if c==4:
        df['grid'] = df['grid']+df['solar']
        df.drop(columns=['solar'])
    if c==5:
        if df['car1'].isnull().all():
            continue
        if df['car1'].isna().all():
            continue
        if df['car1'].empty:
            continue
    df.set_index('localminute', inplace=True)
    dataset = df.values
    dataset = dataset.astype('float32')
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    traind, testd = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    traind = scaler.fit_transform(traind)
    testd = scaler.transform(testd)
    #Keep the real id of the client
    k = filename1.rstrip('.csv')
    k = k.strip('user')
    counter += 1
    groupname = "examples/"+k

  #Turn the data into the supervised learning shape  
    train,test = series_to_supervised(traind, N_LAG, N_SEQ), series_to_supervised(testd, N_LAG, N_SEQ)
    train_values = train.values
    test_values = test.values
    #trainX,trainY= train_values[:, 0:N_LAG], train_values[:, N_LAG:]
    #testX,testY= test_values[:, 0:N_LAG], test_values[:, N_LAG:]
    
    TempGroup = tr.create_group(groupname)
    TempGroup.create_dataset('traindata', data=train_values)
    #TempGroup.create_dataset('realId', data=k)
    TempGroupTs = ts.create_group(groupname)
    TempGroupTs.create_dataset('testdata', data=test_values)
    #TempGroupTs.create_dataset('realId', data=k)
    
    scaler_filename = 'Scalers/'+k+'_1min_'+case+'austin.save'
    joblib.dump(scaler, scaler_filename)
'''  banned = ['user1450','user1524','user203','user2606', 'user3687' ]  
  for filename1 in filenameL2:
    if (filename1 in banned):
        continue
    print('==========================================',filename1)
    dataframe = pd.read_csv(os.path.join(PATH2,filename1), usecols=L, engine='python',infer_datetime_format=True, parse_dates=['localminute'])
    #, index_col=['localminute']
    print(dataframe.head())
    df = dataframe[(dataframe['localminute']>mindate ) & (dataframe['localminute']<maxdate)]
    print(df.head())
    print(df['localminute'].max())
    print(df['localminute'].min())
    if c==4:
        df.drop(columns=['solar'])
    if df.empty:
        continue
    df.set_index('localminute', inplace=True)
    print(df.head())
    dataset = df.values
    dataset = dataset.astype('float32')
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    traind, testd = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    traind = scaler.fit_transform(traind)
    testd = scaler.transform(testd)
    #Keep the real id of the client
    k = filename1.rstrip('.csv')
    k = k.strip('user')
    counter += 1
    groupname = "examples/"+k

  #Turn the data into the supervised learning shape  
    train,test = series_to_supervised(traind, N_LAG, N_SEQ), series_to_supervised(testd, N_LAG, N_SEQ)
    train_values = train.values
    test_values = test.values
    #trainX,trainY= train_values[:, 0:N_LAG], train_values[:, N_LAG:]
    #testX,testY= test_values[:, 0:N_LAG], test_values[:, N_LAG:]
    
    TempGroup = tr.create_group(groupname)
    TempGroup.create_dataset('traindata', data=train_values)
    #TempGroup.create_dataset('realId', data=k)
    TempGroupTs = ts.create_group(groupname)
    TempGroupTs.create_dataset('testdata', data=test_values)
    #TempGroupTs.create_dataset('realId', data=k)
    
    scaler_filename = 'Scalers/'+k+'_'+case+'CA.save'
    joblib.dump(scaler, scaler_filename)'''