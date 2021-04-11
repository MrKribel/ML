from get_data import parser
import os
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb


def show_data(index, data):
    plt.figure(figsize=(18, 9), dpi=100)
    plt.plot(index, data, label='stock')
    plt.xlabel('Time')
    plt.ylabel('Rub')
    plt.title('Figure 1: MTC')
    plt.legend()
    plt.savefig('img/0.png')


def plot_technical_indicators(data, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = data.shape[0]
    xmacd_ = shape_0 - last_days

    dataset = data.iloc[-last_days:, :]

    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)
    # x_ = dataset.time.reset_index(drop=True)

    plt.subplot(2, 1, 1)

    plt.plot(x_, dataset['ma7'].reset_index(drop=True), label='MA 7', color='g', linestyle='--')
    plt.plot(x_, dataset['c'].reset_index(drop=True), label='Closing Price', color='b')
    plt.plot(x_, dataset['ma21'].reset_index(drop=True), label='MA 21', color='r', linestyle='--')
    plt.plot(x_, dataset['upper_band'].reset_index(drop=True), label='Upper Band', color='c')
    plt.plot(x_, dataset['lower_band'].reset_index(drop=True), label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.xticks(rotation=45)
    plt.title('Technical indicators for MTC - last {} days.'.format(last_days))
    plt.ylabel('Rub')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.title('MACD')
    plt.plot(x_, dataset['MACD'].reset_index(drop=True), label='MACD', linestyle='-.')
    plt.plot(x_, dataset['momentum'].reset_index(drop=True), label='Momentum', color='b', linestyle='-')

    plt.hlines(6, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-6, xmacd_, shape_0, colors='g', linestyles='--')

    plt.xticks(rotation=45)
    plt.legend()

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.99,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig('img/1.png')


def load_data():
    data_json = pd.read_json('data/1.json')
    data = json_normalize(data_json['candles']).drop(['figi', 'interval'], axis=1)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M:%SZ')

    return data


def get_technical_indicators(dataset):
    # функция для генерации характерных технических индикаторов

    # 7 и 21  Moving Average
    dataset['ma7'] = dataset['c'].rolling(window=7).mean()
    dataset['ma21'] = dataset['c'].rolling(window=21).mean()

    # MACD
    dataset['26ema'] = dataset['c'].ewm(span=26).mean()
    dataset['12ema'] = dataset['c'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # Bollinger Bands
    dataset['20sd'] = dataset['c'].rolling(window=20).std()
    dataset['upper_band'] = (dataset['c'].rolling(window=20).mean()) + (dataset['20sd'] * 2)
    dataset['lower_band'] = (dataset['c'].rolling(window=20).mean()) - (dataset['20sd'] * 2)

    # Exponential moving average
    dataset['ema'] = dataset['c'].ewm(com=0.5).mean()

    # Momentum
    dataset['momentum'] = (dataset['c'] / 100) - 1

    return dataset


def show_fourie():
    data_FT = data[['time', 'c']]
    close_fft = np.fft.fft(np.asarray(data_FT['c'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data_FT['c'], label='Real')
    plt.xlabel('Days')
    plt.ylabel('Rub')
    plt.title('Figure 3: MTC (close) stock prices & Fourier transforms')
    plt.legend()
    plt.savefig('img/2.png')


def get_fourier(dataset):
    data_FT = dataset[['time', 'c']]
    close_fft = np.fft.fft(np.asarray(data_FT['c'].tolist()))
    close_fft = np.fft.ifft(close_fft)
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_list_m10= np.copy(fft_list); fft_list_m10[100:-100]=0
    dataset['Fourier'] = pd.DataFrame(fft_list_m10).apply(lambda x: np.abs(x))

    return dataset

def ARIMA_model(series):

    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()

    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    dataset_TI_df['ARIMA'] = pd.DataFrame(predictions)

    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(test, label='Real')
    plt.plot(predictions, color='red', label='Predicted')
    plt.xlabel('Days')
    plt.ylabel('Rub')
    plt.title('Figure 5: MTC')
    plt.legend()
    plt.savefig('img/3.png')

    return dataset_TI_df


def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['c']
    X = data.iloc[:, 1:]

    train_samples = int(X.shape[0] * 0.65)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)


def get_feature():
    (X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df)
    regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=200, base_score=0.7, colsample_bytree=1, learning_rate=0.05)
    xgbModel = regressor.fit(X_train_FI._get_numeric_data(), y_train_FI._get_numeric_data(),
                             eval_set=[(X_train_FI._get_numeric_data(), y_train_FI._get_numeric_data()),
                                       (X_test_FI._get_numeric_data(), y_test_FI._get_numeric_data())],
                             verbose=False)

    fig = plt.figure(figsize=(8, 8))

    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(),
            tick_label=X_test_FI._get_numeric_data().columns)
    plt.title('Figure 6: Feature importance of the technical indicators.')
    plt.savefig('img/4.png')


def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
        a = dataset[i:(i+step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

def prepering_data(dataset_TI_df):
    dataset = dataset_TI_df[['o', 'c', 'h', 'l']]

    dataset = dataset.reindex(index = dataset.index[::-1])

    dataset['OHLC_avg'] = dataset[['o', 'c', 'h', 'l']].mean(axis=1)
    dataset['HLC_avg'] = dataset[['h', 'l', 'c']].mean(axis=1)

    OHLC_avg = np.reshape(dataset.OHLC_avg.values, (len(dataset.OHLC_avg), 1))  # переворачиваем
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg_scal = scaler.fit_transform(OHLC_avg)  # нормализуем

    train_OHLC_len = int(len(dataset.OHLC_avg) * 0.75)
    test_OHLC_len = len(dataset.OHLC_avg) - train_OHLC_len

    train_OHLC, test_OHLC = OHLC_avg_scal[0:train_OHLC_len, :], OHLC_avg_scal[train_OHLC_len:len(OHLC_avg_scal), :]



    trainX, trainY = new_dataset(train_OHLC, 1)
    testX, testY = new_dataset(test_OHLC, 1)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, testX, trainY, testY, scaler, OHLC_avg



def build_LSTM(trainX, trainY):
    step_size=1

    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, step_size), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    return model

def predict_LSTM(model, trainX, testX, trainY, testY, scaler, OHLC_avg):
    step_size=1

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])


    # Создаем датасет для отрисовки тренировочных предсказаний
    trainPredictPlot = np.empty_like(OHLC_avg)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

    # Создаем датасет для отрисовки тестовых предсказаний
    testPredictPlot = np.empty_like(OHLC_avg)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (step_size * 2) + 1:len(OHLC_avg) - 1, :] = testPredict

    # Отображаем OHLC значения на графике, тренировочные предсказанные и тестовые предсказанные
    with plt.style.context('bmh'):
        plt.figure(figsize=(18, 8))
        plt.plot(trainPredictPlot, 'r', label='training set')
        plt.plot(testPredictPlot, 'b', label='predicted stock price/test set')
        plt.legend(loc='upper right')
        plt.xlabel('Time in Days')
        plt.ylabel('Trend of training and prediction data')
        plt.savefig('img/5.png')

if __name__ == "__main__":
    if os.path.isfile('data/1.json'):
        pass
    else:
        parser(os.environ['login'], os.environ['password'])
    data = load_data()

    show_data(data.index, data['c'])

    dataset_TI_df = get_technical_indicators(data)

    plot_technical_indicators(dataset_TI_df, 200)

    show_fourie()

    dataset_TI_df = get_fourier(data)

    dataset_TI_df = ARIMA_model(dataset_TI_df['c'])
    get_feature()
    trainX, testX, trainY, testY, scaler, OHLC_avg = prepering_data(dataset_TI_df)

    model = build_LSTM(trainX, trainY)

    predict_LSTM(model, trainX, testX, trainY, testY, scaler, OHLC_avg)








