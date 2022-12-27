import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.gradient_descent import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
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

def FFmodel(learning_rate=0.01, momentum=0.1):
    model = Sequential()
    model.add(Dense(64, input_dim=seq_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    optimizer = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])
    return model

Data_path = "C:/Users/ideapad/Desktop/Luk_work/TUxSA/IS/weather data/weather_training_data_0921_0322.xlsx"
future = 12
alldata = pd.read_excel(Data_path, sheet_name='model data every 30 min').iloc[0:13903-future] #test train till date before future else for validate till 13903 (220616_1500)
Traget = ((alldata['Out Hum'].values).astype('float32')).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(Traget)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

seq_size = 4
trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)

batch_size = 100
epochs = 100

model = KerasRegressor(build_fn=FFmodel, epochs=epochs,batch_size = batch_size,verbose=1)

learning_rate = [0.0001, 0.001, 0.01, 0.1]
momentum = [0.3, 0.5, 0.7, 0.9]

param_grid = dict(learning_rate=learning_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=8, cv=3, scoring='neg_root_mean_squared_error')
grid_result = grid.fit(trainX, trainY)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = (grid_result.cv_results_['mean_test_score'])*-1
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean = %f (std=%f) with: %r" % (mean, stdev, param))




# #--- counf plot
# result_array = np.zeros([3, len(params)])
# for i in range(0, len(params)):
#     result_array[0, i] = list(params[i].values())[0]
#     result_array[1, i] = list(params[i].values())[1]
# result_array[2] = means
# X = np.reshape(result_array[0], (len(learning_rate), len(momentum)))
# Y = np.reshape(result_array[1], (len(learning_rate), len(momentum)))
# Z = np.reshape(result_array[2], (len(learning_rate), len(momentum)))
# plt.contourf(X, Y, Z, 50, cmap='plasma')
# plt.colorbar()
# plt.show()

