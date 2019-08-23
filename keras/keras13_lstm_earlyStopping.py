from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape :",x.shape)
print("y.shape :",y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))

# 2.모델 구성
model = Sequential()
model.add(LSTM(500, activation = 'relu', input_shape=(3,1)))     # 3열 1개씩 ,4(nm + n^2 +n)
model.add(Dense(30))
# model.add(Dense(25))
# model.add(Dense(20))
# model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1))

model.add(Dense(1))


model.summary()

#3. 실행
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
hist = model.fit(x, y, epochs=10000, batch_size=3, verbose=1, 
                callbacks=[early_stopping])

#4. 평가 및 예측
#x_input = array([11,12,13])    # 1행 3열 , 잘라서 하는 작업 크기 필요
x_input = array([25,35,45])    # 1행 3열 , 잘라서 하는 작업 크기 필요
print(x_input.shape)
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input, verbose=2)
print(yhat)