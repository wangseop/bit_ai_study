import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np

import matplotlib.pyplot as plt

# RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))



# 1. 데이터
data = pd.read_csv("kospi200test.csv", encoding='euc-kr')
# 일자       시가       고가       저가       종가      거래량  환율(원/달러)
NUM_EPOCHS=100
NUM_BATCH_SIZE = 1
size = 4

# x_arr = data.as_matrix(columns=['시가', '고가', '저가', '종가', '거래량'])
x_arr = data.as_matrix(columns=['시가', '고가', '저가', '종가'])

y_arr = data.as_matrix(columns=['종가'])

x_arr = np.flip(x_arr)
y_arr = np.flip(y_arr)

# Scaler 적용
scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_arr)
trans_x_arr = scaler.transform(x_arr)

# 데이터 split
dataX = []
dataY = []

for i in range(0, len(trans_x_arr) - size - 1):
    _x = trans_x_arr[i:i + size]
    _y = y_arr[i + size:i+size+2]
    dataX.append(_x)
    dataY.append(_y)

# train, val, test 분할
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    dataX, dataY, random_state=66, test_size = 0.8
)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1] * y_train.shape[2]))
y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1] * y_val.shape[2]))
x_test = np.array(dataX[-1:])
# 2. 모델

def kospiLSTM():
    model = Sequential()
    # model.add(LSTM(12, batch_input_shape=(NUM_BATCH_SIZE, size, 4), stateful=True, return_sequences=True))
    # model.add(LSTM(14))
    # model.add(Dense(40))
    # model.add(Dense(50))
    # model.add(Dense(2))
    model.add(LSTM(4, batch_input_shape=(NUM_BATCH_SIZE, size, 4), stateful=True))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    return model

# 3. 훈련

model = kospiLSTM()
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

epochs_cnt = 5
histories = []
for epoch_idx in range(epochs_cnt):
    print('epochs :', epoch_idx)
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=NUM_BATCH_SIZE, verbose=1, callbacks=[early_stopping], validation_data=(x_val, y_val))

    model.reset_states() 
    histories.append(history)

# 4. 평가 및 예측
mse, _ = model.evaluate(x_val, y_val, batch_size=NUM_BATCH_SIZE)

print('mse :', mse)
model.reset_states() 

y_predict = model.predict(x_test, batch_size=NUM_BATCH_SIZE)
print(y_predict)
# RMSE 함수 적용
# rmse = RMSE(y_test, y_predict)

# R2 함수 적용
# y_r2 = r2_score(y_test, y_predict)


