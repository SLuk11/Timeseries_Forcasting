from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def Univariate_f_reg(Xs, Traget, Nfeature):
    selector = SelectKBest(f_regression, k=Nfeature)
    selector.fit_transform(Xs, Traget)
    cols = selector.get_support(indices=True)
    Xs_new = Xs.iloc[:, cols]
    print('---  Univariate Selection (f_regression) ---')
    print(Xs_new)
    print(' ')
    return

def Univariate_mutual_reg(Xs, Traget, Nfeature):
    selector = SelectKBest(mutual_info_regression, k=Nfeature)
    selector.fit_transform(Xs, Traget)
    cols = selector.get_support(indices=True)
    Xs_new = Xs.iloc[:, cols]
    print('--- Univariate Selection (mutual_info_regression) ---')
    print(Xs_new)
    print(' ')
    return

def RFE_linear(Xs, Traget, Nfeature):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=Nfeature, step=1)
    selector = selector.fit(Xs, Traget)
    cols = selector.get_support(indices=True)
    score = selector.ranking_
    Xs_new = Xs.iloc[:, cols]
    print('--- Recursive feature elimination (RFE) ---')
    print(Xs.columns)
    print(score)
    print(' ')
    return

def Impotance_LinearReg(Xs, Traget):
    Xs_array = Xs.values
    Traget_array = Traget.values
    X = Xs_array[:,0:Xs_array.shape[1]]
    Y = Traget_array
    model = LinearRegression()
    model.fit(X, Y)
    importance = model.coef_
    print('--- Feature Importance (Linear Regression) ---')
    for i, v in enumerate(importance):
        print('Feature: ' + str(Xs.columns[i]) + ', Score: %.5f' % (v))
    print(' ')
    return

def Impotance_CART(Xs, Traget):
    Xs_array = Xs.values
    Traget_array = Traget.values
    X = Xs_array[:,0:Xs_array.shape[1]]
    Y = Traget_array
    model = DecisionTreeRegressor()
    model.fit(X, Y)
    importance = model.feature_importances_
    print('--- Feature Importance (Classification and regression trees (CART)) ---')
    for i, v in enumerate(importance):
        print('Feature: ' + str(Xs.columns[i]) + ', Score: %.5f' % (v))
    print(' ')
    return

Data_path = "C:/Users/ideapad/Desktop/Luk_work/TUxSA/IS/weather data/weather_training_data_0921_0322.xlsx"
alldata = pd.read_excel(Data_path, sheet_name='model data every30min').iloc[0:10178]
DateTime = pd.DataFrame(alldata["Date"].astype(str)+ "_" + alldata["Time"].astype(str), columns=['DateTime'])
Data = (alldata.drop(['Date','Time'], axis=1)).astype(float)
scaler = StandardScaler()
data_scale = scaler.fit_transform(Data)
Traning_data = pd.DataFrame(data_scale, columns=Data.columns)
Traning_data_w0_OutHum = Traning_data.drop(['Out Hum'], axis=1)
Traget = Traning_data['Out Hum']

for i in range(1,5):
    Univariate_f_reg(Traning_data_w0_OutHum, Traget, i)
    Univariate_mutual_reg(Traning_data_w0_OutHum, Traget, i)

RFE_linear(Traning_data, Traget, 1)
Impotance_LinearReg(Traning_data_w0_OutHum, Traget)
Impotance_CART(Traning_data_w0_OutHum, Traget)
