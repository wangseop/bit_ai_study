# 1 ~100 까지의 숫자를 이용해서
# 6개씩 잘라서 rnn 구성
# train, test 분리할 것

# 1,2,3,4,5,6 : 7
# 2,3,4,5,6,7 : 8
# 3,4,5,6,7,8 : 9
# ...
# 94,95,96,97,98,99 : 100

# predict : 101 ~ 110 까지 예측하시오.
# 지표 rmse

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

## constant
SPLIT_SIZE = 6

## function
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def func_split_data(data, size):
    x_split_data = []
    for i in range(len(data) - size + 1):
        split = data[i:i+size]
        x_split_data.append(split)
    return np.array(x_split_data)

## ##

data = np.array(np.arange(1, 101))

print(data) # 1 ~ 100
print(data.shape)   # (100, )


split_data = func_split_data(data, SPLIT_SIZE+1)
x_data = split_data[:, :-1]
y_data = split_data[:, [-1]]

print(x_data)       
print(x_data.shape) # (94, 6)
print(y_data)
print(y_data.shape) # (94, )


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2
)

x_train = x_train.reshape(-1, SPLIT_SIZE, 1)
x_test = x_test.reshape(-1, SPLIT_SIZE, 1)


model = Sequential()
model.add(LSTM(128, input_shape=(6,1)))

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))


model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])
model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=1)

## test로 평가

_, mse, acc = model.evaluate(x_test, y_test, batch_size=32)
y_pred = model.predict(x_test)
## 값 출력
print("evaluate mse :", mse)
print("evaluate acc :", acc)

print("y_test_values : ", y_test)
print("y_pred_values : ", y_pred)
print("rmse :", RMSE(y_test, y_pred))

target_data = np.array(np.arange(95,111))
target_split_data = func_split_data(target_data, SPLIT_SIZE+1)
x_target = target_split_data[:, :-1]
y_target = target_split_data[:, [-1]]

x_target = x_target.reshape(-1, SPLIT_SIZE, 1)
y_pred2 = model.predict(x_target)
print("y_target_values : ", y_target)
print("y_pred2_values : ", y_pred2)
print("RMSE :", RMSE(y_target, y_pred2))
