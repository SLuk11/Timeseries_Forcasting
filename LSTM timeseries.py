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

def to_sequences(dataset,lookback):
    x = []
    y = []
    for j in range(len(dataset)-lookback):
        window = dataset[j:(j+lookback),0]
        x.append(window)
        y.append(dataset[j+lookback,0])

    return np.array(x), np.array(y)

Data_path = "__________________________________.xlsx"
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
col_set = alldata[{'Out Hum'}]
Input = pd.concat([col_set], axis=1)
first_column = Input.pop('Out Hum')
Input.insert(0, 'Out Hum', first_column)

lookback_step = 72
train_size = int(len(Input)*0.75)
test_size = len(Input) - train_size
train = Input.iloc[0:train_size]
test = Input.iloc[train_size:]

scaler = StandardScaler()
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

train_oh_x, train_oh_y = to_sequences(train_scaled , lookback_step)
test_oh_x, test_oh_y = to_sequences(test_scaled, lookback_step)
train_oh_x = np.reshape(train_oh_x, (train_oh_x.shape[0], 1, train_oh_x.shape[1]))
test_oh_x = np.reshape(test_oh_x, (test_oh_x.shape[0], 1, test_oh_x.shape[1]))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(None, lookback_step), return_sequences=True))
model.add(LSTM(32, activation='relu',return_sequences=False))
model.add(Dense(1))
optimizer = SGD(lr=0.1, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['acc'])
model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=30,restore_best_weights=True),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(train_oh_x, train_oh_y, epochs=100, callbacks=callbacks, batch_size=100, validation_split=0.3, verbose=2)
# model.fit(train_oh_x, train_oh_y, validation_data=(test_oh_x , test_oh_y), verbose=2, epochs=100)
trainPredict = model.predict(train_oh_x)
trainPredict_inv = scaler.inverse_transform(trainPredict)
trainPredict_inv = np.delete(trainPredict_inv, 0)
trainY_inv = scaler.inverse_transform(train_oh_y)
trainY_inv = np.delete(trainY_inv, -1)

testPredict = model.predict(test_oh_x)
testPredict_inv = scaler.inverse_transform(testPredict)
testPredict_inv = np.delete(testPredict_inv, 0)
testY_inv = scaler.inverse_transform(test_oh_y)
testY_inv = np.delete(testY_inv, -1)

trainScore = math.sqrt(mean_squared_error(trainY_inv, trainPredict_inv))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_inv, testPredict_inv))
print('Test Score: %.2f RMSE' % (testScore))

#Train Test Plot
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
prediction = [] #Empty list to populate later with predictions
current_batch = test_scaled[-lookback_step:] #Final data points in dataset
current_batch = current_batch.reshape(1, 1,lookback_step) #Reshape

for i in range(future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,:,1:],[[current_pred]],axis=2)
prediction_inverse = scaler.inverse_transform(prediction)
#prediction_inverse = np.delete(prediction_inverse, -1)

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

futurePredictPlot = pd.DataFrame(prediction_inverse, columns=['Out Hum'])
futurePredictPlot.index = F_data.index

predScore = math.sqrt(mean_squared_error(F_data, futurePredictPlot))
print('Prediction Score: %.2f RMSE' % (predScore))
print("Future Prediction")
print(futurePredictPlot)

plt.plot(F_data)
plt.plot(futurePredictPlot)
plt.show()
