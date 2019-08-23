import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D

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
NUM_EPOCHS=1000
NUM_BATCH_SIZE = 1
size = 28

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
for i in range(0, len(trans_x_arr) - size - 14):    # 14일치 미래 예측이므로 13일치 만큼 빼준다
    _x = trans_x_arr[i:i + size]
    _y = y_arr[i + size:i+size+14]
    dataX.append(_x)
    dataY.append(_y)

# train, val, test 분할
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    dataX, dataY, random_state=66, test_size = 0.2
)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)
x_val = np.array(x_val)
y_val = np.array(y_val)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1] * y_train.shape[2]))
y_val = np.reshape(y_val, (y_val.shape[0],y_val.shape[1] * y_val.shape[2]))
x_test = np.array(dataX[-1:])
# 2. 모델

def kospiLSTM():
    model = Sequential()

    model.add(Conv1D(15, kernel_size=7, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
    model.add(Conv1D(16, kernel_size=7, padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Conv1D(20, kernel_size=6, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
    model.add(Conv1D(22, kernel_size=6, padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Conv1D(27, kernel_size=5, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
    model.add(Conv1D(29, kernel_size=5, padding='same'))
    model.add(Conv1D(32, kernel_size=5, padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Conv1D(37, kernel_size=4, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
    model.add(Conv1D(40, kernel_size=4, padding='same'))
    model.add(Conv1D(41, kernel_size=4, padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(14))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    return model

# 3. 훈련

model = kospiLSTM()
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

epochs_cnt = 5
histories = []
# for epoch_idx in range(epochs_cnt):
#     print('epochs :', epoch_idx)
#     history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=NUM_BATCH_SIZE, verbose=1, callbacks=[early_stopping], validation_data=(x_val, y_val))

#     model.reset_states() 
#     histories.append(history)
history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=NUM_BATCH_SIZE, verbose=1, callbacks=[early_stopping], validation_data=(x_val, y_val))

# 4. 평가 및 예측
mse, _ = model.evaluate(x_val, y_val, batch_size=NUM_BATCH_SIZE)

print('mse :', mse)
# model.reset_states() 

y_predict = model.predict(x_test, batch_size=NUM_BATCH_SIZE)
print(y_predict)
# RMSE 함수 적용
# rmse = RMSE(y_test, y_predict)

# R2 함수 적용
# y_r2 = r2_score(y_test, y_predict)


"""
model = Sequential()

model.add(Conv1D(13, kernel_size=7, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(15, kernel_size=7, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(16, kernel_size=6, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(18, kernel_size=6, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(21, kernel_size=5, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(23, kernel_size=5, padding='same'))
model.add(Conv1D(26, kernel_size=5, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(21, kernel_size=4, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(23, kernel_size=4, padding='same'))
model.add(Conv1D(26, kernel_size=4, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(14))

mse : 2923.3735448292323
[[2028.8989 2038.0997 2051.3901 2048.0605 2063.416  2066.6526 2056.0867
  2056.9712 2044.8627 2040.6405 2016.5704 2025.9995 2017.2474 2014.7968]]
"""


"""
model = Sequential()

model.add(Conv1D(13, kernel_size=7, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(15, kernel_size=7, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(16, kernel_size=6, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(18, kernel_size=6, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(21, kernel_size=5, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(23, kernel_size=5, padding='same'))
model.add(Conv1D(26, kernel_size=5, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Conv1D(29, kernel_size=4, padding='same', batch_input_shape=(NUM_BATCH_SIZE, size, 4)))
model.add(Conv1D(33, kernel_size=4, padding='same'))
model.add(Conv1D(36, kernel_size=4, padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(66))
model.add(Dense(14))

mse : 1326.46713747297
[[2086.4019 2089.5063 2083.634  2084.5596 2084.3877 2080.3403 2104.006
  2114.8613 2119.422  2116.7869 2094.645  2093.753  2092.657  2090.7732]]
"""

"""
mse : 1055.4132540566582
[[2044.2856 2052.243  2066.1106 2072.168  2059.6167 2058.0933 2062.9595
  2062.148  2064.016  2076.927  2073.063  2061.4385 2065.9546 2058.0562]]
"""