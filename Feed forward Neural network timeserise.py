import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizer_v2.gradient_descent import SGD
import pandas as pd


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size - 1):
        # print(i)
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)

Data_path = "C:/Users/ideapad/Desktop/Luk_work/TUxSA/IS/weather data/weather_training_data_0921_0322.xlsx"
future = 12
alldata = pd.read_excel(Data_path, sheet_name='model data every 30 min').iloc[0:13903-future] #test train till date before future else for validate till 13903 (220616_1500)
Traget = ((alldata['Out Hum'].values).astype('float32')).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(Traget)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

seq_size = 72
trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)

model = Sequential()
model.add(Dense(64, input_dim=seq_size, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
optimizer = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['acc'])
callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(trainX, trainY, epochs=100, callbacks=callbacks, batch_size=100, validation_split=0.3, verbose=2)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = np.delete(trainPredict, 0)
trainY_inverse = scaler.inverse_transform([trainY]).transpose()
trainY_inverse = np.delete(trainY_inverse, -1)

testPredict = scaler.inverse_transform(testPredict)
testPredict = np.delete(testPredict, 0)
testY_inverse = scaler.inverse_transform([testY]).transpose()
testY_inverse = np.delete(testY_inverse, -1)

trainScore = math.sqrt(mean_squared_error(trainY_inverse, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_inverse, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

#Train Test Pot
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict.reshape(-1,1)

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+2:len(trainPredict)+(seq_size*2)+2+len(testPredict), :] = testPredict.reshape(-1,1)

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

#forecast
prediction = [] #Empty list to populate later with predictions
current_batch = dataset[-seq_size:] #Final data points in dataset
current_batch = current_batch.reshape(1, seq_size,1) #Reshape

for i in range(future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
prediction_inverse = scaler.inverse_transform(prediction)
prediction_inverse = np.delete(prediction_inverse, -1)

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