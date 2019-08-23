import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout

# X = np.array(range(1, 101))
# Y = np.array(range(1, 101))

# size = 4
# def split_size(seq, size):
#     aaa = []
#     for i in range(len(seq) - size + 1):
#         subset = seq[i:(i+size)]
#         aaa.append([item for item in subset])
#     print(type(aaa))
#     return np.array(aaa)

# x_split = split_size(X, size)
# y_split = split_size(Y, size)

# x_train = x_split[:93]
# y_train = y_split[4:]

# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(
#     x_train, y_train, random_state=66, test_size = 0.4
# )
# x_test = x_split[1:5]
# y_test = x_test + 4

X = np.array(range(1, 101))

size = 8
def split_size(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

data_split = split_size(X, size)

x_train = data_split[:,:4]
y_train = data_split[:,4:]


x_train = np.reshape(x_train, (x_train.shape[0], size//2, 1))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, random_state=66, test_size = 0.2
)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

model = Sequential()

model.add(LSTM(30, input_shape=(4,1)))
# model.add(LSTM(8))
model.add(Dense(8, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))

model.add(Dense(4))

model.summary()

# 3.훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss :', loss)
print('y_test :', y_test)
print('y_predict(x_test) :', y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 :', r2_y_predict)

