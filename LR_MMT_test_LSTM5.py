import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizer_v2.gradient_descent import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler


def to_sequences_mulit(dataset, lookback,n_future = 1):
    x = []
    y = []
    for i in range(lookback, len(dataset)-n_future+1):
        x.append(dataset[i - lookback:i, 0:dataset.shape[1]])
        y.append(dataset[i + n_future - 1:i + n_future,0])
    return np.array(x), np.array(y)

def LSTM5model(learning_rate=0.01, momentum=0.1):
    model = Sequential()
    model.add(
        LSTM(64, activation='relu', input_shape=(train_scaled_X.shape[1], train_scaled_X.shape[2]),
             return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dense(train_scaled_Y.shape[1]))
    optimizer = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])
    return model

Data_path = "C:/Users/ideapad/Desktop/Luk_work/TUxSA/IS/weather data/weather_training_data_0921_0322.xlsx"
future = 12
alldata = pd.read_excel(Data_path, sheet_name='model data every 30 min').iloc[0:13903-future] #test train till date before future else for validate till 13903 (220616_1500)
Data = alldata.drop(['Date','Time'], axis=1)
Date = pd.DataFrame(alldata['Date'].astype(str), columns=['Date'])
Time = pd.DataFrame(alldata['Time'].astype(str), columns=['Time'])
Date.Date.str.replace('/', '-')
DT_comb = Date.Date.str.cat(Time.Time, sep=' ')
Traindata = pd.concat([DT_comb, Data], axis=1)
Traindata['Date'] = pd.to_datetime(Traindata['Date'])
Traindata.set_index('Date', inplace=True)
col_set = alldata[{'Out Hum', 'Dew Pt.', 'Tmp Out', 'Wind Chill', 'THW Index'}] # Recursive feature elimination (RFE) set

Input = pd.concat([col_set], axis=1)
first_column = Input.pop('Out Hum')
Input.insert(0, 'Out Hum', first_column)

lookback_step = 4
train_size = int(len(Input)*0.75)
test_size = len(Input) - train_size
train = Input.iloc[0:train_size]
test = Input.iloc[train_size:]

scaler = StandardScaler()
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
n_past = lookback_step

train_scaled_X, train_scaled_Y = to_sequences_mulit(train_scaled, n_past)
test_scaled_X, test_scaled_Y = to_sequences_mulit(test_scaled, n_past)
print('trainX shape == {}'.format(train_scaled_X.shape))
print('trainY shape == {}'.format(train_scaled_Y.shape))
print('testX shape == {}'.format(test_scaled_X.shape))
print('testY shape == {}'.format(test_scaled_Y.shape))

batch_size = 100
epochs = 100

model = KerasRegressor(build_fn=LSTM5model, epochs=epochs,batch_size = batch_size,verbose=1)

learning_rate = [0.0001, 0.001, 0.01, 0.1]
momentum = [0.3, 0.5, 0.7, 0.9]

param_grid = dict(learning_rate=learning_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error')
grid_result = grid.fit(train_scaled_X, train_scaled_Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = (grid_result.cv_results_['mean_test_score'])*-1
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean = %f (std=%f) with: %r" % (mean, stdev, param))