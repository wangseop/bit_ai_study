#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 모델링 과정
model.add(Dense(5, input_dim = 1, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
# model.add(Dense(10))
# model.add(Dense(15))
# model.add(Dense(20))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('acc : ', acc)

y_predict = model.predict(x2)       # 새로운 값 예측
print(y_predict)
