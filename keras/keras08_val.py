#1. 데이터
import numpy as np
x_train = np.arange(1, 11, 1)
y_train = np.arange(1, 11, 1)
x_test = np.arange(11, 21, 1)
y_test = np.arange(11, 21, 1)
x_val = np.array(range(101, 106))
y_val = np.array(range(101, 106))

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense  
model = Sequential()        # 순차적인 모델

# 모델링 과정
# model.add(Dense(2, input_dim = 1, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(1000, input_shape = (1, ), activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#model.fit(x, y, epochs=100)
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data= (x_val, y_val))

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)    # 평가 지표
print('acc : ', acc)

y_predict = model.predict(x_test)       # 새로운 값 예측
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 :', r2_y_predict)

# RMAE 구하기
from sklearn.metrics import mean_absolute_error
def RMAE(y_test, y_predict):
    return mean_absolute_error(y_test, y_predict)
# print('RMAE :', RMAE(y_test, y_predict))

# 실습 R2를 음수가 아닌 0.5 이하로 줄이기
# 레이어는 인풋과아웃풋을 포함 5개이상, 노드는 5개이상
# batch_size = 1
# epochs = 100 이상