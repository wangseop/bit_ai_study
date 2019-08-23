#1. 데이터
import numpy as np
x_train = np.arange(1, 11, 1)
y_train = np.arange(1, 11, 1)
x_test = np.arange(11, 21, 1)
y_test = np.arange(11, 21, 1)
x3 = np.arange(101, 107, 1)
x4 = np.array(range(30,50))

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential()        # 순차적인 모델

# 모델링 과정
# model.add(Dense(2, input_dim = 1, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(2, input_shape = (1, ), activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=200, batch_size=3)
model.fit(x_train, y_train, epochs=800, batch_size=3)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print('acc : ', acc)

y_predict = model.predict(x4)       # 새로운 값 예측
print(y_predict)
