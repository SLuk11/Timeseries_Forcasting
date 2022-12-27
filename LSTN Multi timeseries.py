import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.optimizer_v2.gradient_descent import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math

def to_sequences_mulit(dataset, lookback,n_future = 1):
    x = []
    y = []
    for i in range(lookback, len(dataset)-n_future+1):
        x.append(dataset[i - lookback:i, 0:dataset.shape[1]])
        y.append(dataset[i + n_future - 1:i + n_future,0])
    return np.array(x), np.array(y)


Data_path = "________________________________.xlsx"
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
# col_set = alldata[{'Out Hum', 'Dew Pt.', 'Lo Temp_forest', 'Lo Temp_tmd', 'glass_temp'}] # Univariate Selection (f_regression) set
# col_set = alldata[{'Out Hum', 'Dew Pt.', 'glass_temp', 'Hi Temp_tmd', 'Lo Temp_tmd'}] # Univariate Selection (mutual_info_regression) set
# col_set = alldata[{'Out Hum', 'THW Index', 'Heat Index', 'Low Temp', 'Dew Pt.'}] # Feature Importance (Linear Regression) set
# col_set = alldata[{'Out Hum', 'Dew Pt.', 'Hi Temp', 'Heat D-D', 'Tmp Out'}] # Feature Importance (Classification and regression trees (CART)) set

Input = pd.concat([col_set], axis=1)
first_column = Input.pop('Out Hum')
Input.insert(0, 'Out Hum', first_column)

lookback_step = 28
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

model = Sequential()
model.add(
    LSTM(64, activation='relu', input_shape=(train_scaled_X.shape[1], train_scaled_X.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(train_scaled_Y.shape[1]))
optimizer = SGD(lr=0.1, momentum=0.3)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['acc'])
model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=30,restore_best_weights=True),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(train_scaled_X, train_scaled_Y, epochs=100, callbacks=callbacks, batch_size=100,
                    validation_split=0.3, verbose=2)

trainPredict = model.predict(train_scaled_X)
trainPredict_copies = np.repeat(trainPredict, train.shape[1], axis=-1)
trainPredict_inv = scaler.inverse_transform(trainPredict_copies)[:, 0]
trainPredict_inv = np.delete(trainPredict_inv, 0)
train_scaled_Y_copies = np.repeat(train_scaled_Y, train.shape[1], axis=-1)
trainY_inv = scaler.inverse_transform(train_scaled_Y_copies)[:, 0]
trainY_inv = np.delete(trainY_inv, -1)

testPredict = model.predict(test_scaled_X)
testPredict_copies = np.repeat(testPredict, train.shape[1], axis=-1)
testPredict_inv = scaler.inverse_transform(testPredict_copies)[:, 0]
testPredict_inv = np.delete(testPredict_inv, 0)
test_scaled_Y_copies = np.repeat(test_scaled_Y, train.shape[1], axis=-1)
testY_inv = scaler.inverse_transform(test_scaled_Y_copies)[:, 0]
testY_inv = np.delete(testY_inv, -1)

trainScore = math.sqrt(mean_squared_error(trainY_inv, trainPredict_inv))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_inv, testPredict_inv))
print('Test Score: %.2f RMSE' % (testScore))

# Train Test Plot
trainPredictPlot = np.empty_like(Traindata)
trainPredictPlot[:, :] = np.nan
train_prediction_inv = trainPredict_inv.reshape(-1,1)
trainPredictPlot[lookback_step:lookback_step + len(train_prediction_inv), :] = train_prediction_inv
dfTrain = pd.concat([DT_comb, pd.Series(trainPredictPlot[:,0])], axis=1)
dfTrain['Date'] = pd.to_datetime(dfTrain['Date'])
dfTrain.set_index('Date', inplace=True)

testPredictPlot = np.empty_like(Traindata)
testPredictPlot[:, :] = np.nan
test_prediction_inv = testPredict_inv.reshape(-1,1)
testPredictPlot[len(train) + lookback_step:len(train)+ lookback_step+len(test_prediction_inv), :] = test_prediction_inv
dfTest = pd.concat([DT_comb, pd.Series(testPredictPlot[:,0])], axis=1)
dfTest['Date'] = pd.to_datetime(dfTest['Date'])
dfTest.set_index('Date', inplace=True)

plt.plot(Traindata['Out Hum'])
plt.plot(dfTrain)
plt.plot(dfTest)
plt.show()

#forecast
arr_past=test_scaled[-n_past:,:]
rep_last = np.repeat([arr_past[-1]], future, axis = 0)
future_input = np.vstack([arr_past, rep_last])
future_input[-future:,0] = np.nan
all_data=[]
for j in range(n_past,len(future_input)):
    data_x=[]
    data_x.append(future_input[j-n_past :j , 0:future_input.shape[1]])
    data_x=np.array(data_x)
    prediction=model.predict(data_x)
    all_data.append(prediction)
    future_input[j,0] = prediction
f_pred_array=np.array(all_data)
f_pred_array=f_pred_array.reshape(-1,1)
futurePredict_copies = np.repeat(f_pred_array, train.shape[1], axis=-1)
futurePredict_inv = scaler.inverse_transform(futurePredict_copies)[:, 0]

future_data = pd.read_excel(Data_path, sheet_name='model data every 30 min').iloc[13903-future:13903]
Data_f = future_data.drop(['Date','Time'], axis=1)
Date_f = pd.DataFrame(future_data['Date'].astype(str), columns=['Date'])
Time_f = pd.DataFrame(future_data['Time'].astype(str), columns=['Time'])
Date_f.Date.str.replace('/', '-')
DT_comb_f = Date_f.Date.str.cat(Time_f.Time, sep=' ')
fdata = pd.concat([DT_comb_f, Data_f], axis=1)
fdata['Date'] = pd.to_datetime(fdata['Date'])
fdata.set_index('Date', inplace=True)
f_col_set = future_data[{'Out Hum'}]
F_data = pd.concat([f_col_set], axis=1)
F_data.index = fdata.index

futurePredictPlot = pd.DataFrame(futurePredict_inv, columns=['Out Hum'])
futurePredictPlot.index = F_data.index

predScore = math.sqrt(mean_squared_error(F_data, futurePredictPlot))
print('Prediction Score: %.2f RMSE' % (predScore))
print("Future Prediction")
print(futurePredictPlot)

plt.plot(F_data)
plt.plot(futurePredictPlot)
plt.show()
