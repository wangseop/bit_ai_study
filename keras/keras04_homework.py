#1. 데이터
import numpy as np

# x_train_list = []
# y_train_list = []
# x_test_list = []
# y_test_list = []

# for i in range(1, 101):
#     x_train_list.append(i)
#     y_train_list.append(i+500)
#     x_test_list.append(i+1000)
#     y_test_list.append(i+1100)

# x_train = np.array(x_train_list)
# y_train = np.array(y_train_list)
# x_test = np.array(x_test_list)
# y_test = np.array(y_test_list)

x_train = np.arange(1, 101, 1)
y_train = np.arange(501, 601, 1)
x_test = np.arange(1001, 1101, 1)
y_test = np.arange(1101, 1201, 1)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()        # 순차적인 모델

# 모델링 과정
model.add(Dense(50, input_dim = 1, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=200, batch_size=3)
model.fit(x_train, y_train, epochs=800, batch_size=1)
model.fit

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)

# y_predict = model.predict(x_train)       # 새로운 값 예측
y_predict = model.predict(x_test)       # 새로운 값 예측
print(y_predict)
