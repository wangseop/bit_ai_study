from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
            [5,6,7], [6,7,8], [7,8,9], [8,9,10],
            [9,10,11], [10,11,12], [90,91,92]])

y = array([4,5,6,7,8,9,10,11,12,13,93])

print("x.shape :",x.shape)
print("y.shape :",y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))

# 2.모델 구성
model = Sequential()
model.add(LSTM(22, activation = 'relu', input_shape=(3,1)))     # 3열 1개씩 ,4(nm + n^2 +n)
model.add(Dense(15))
model.add(Dense(16))
model.add(Dense(17))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=3)

#4. 평가 및 예측
#x_input = array([11,12,13])    # 1행 3열 , 잘라서 하는 작업 크기 필요
x_input = array([[11,12,13], [70,80,90]])    # 1행 3열 , 잘라서 하는 작업 크기 필요
print(x_input.shape)
x_input = x_input.reshape((2,3,1))

yhat = model.predict(x_input)
print(yhat)