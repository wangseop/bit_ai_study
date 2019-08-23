import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

a = np.array(range(1,101))
batch_size = 1
size = 5

def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
dataset = split_5(a, size)
print('========================')
print(dataset)
print(dataset.shape)

x_train = dataset[:,0:4]
y_train = dataset[:,4]

x_train = np.reshape(x_train, (len(x_train), size-1, 1))

x_test = x_train + 100
y_test = y_train + 100

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])

# 2. 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(batch_size,4,1), stateful=True))
# └stateful LSTM은 기존 LSTM과 다르게 상태유지를 해주며, 정확도가 높다
# └batch_input_shape는 input_shape에 batch_size를 결합해준 것으로 (batch_size, cols, cutting_size) 로 구성한다
model.add(Dense(200))
model.add(Dense(210))
model.add(Dense(220))
model.add(Dense(225))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

num_epochs = 100

# EarlyStopping 정의
earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=20)
tb_hist = TensorBoard(log_dir='./graph',
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)
histories=[]
for epoch_idx in range(num_epochs):
    print('epochs :', epoch_idx)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=100, verbose=2, shuffle=False,
                validation_data=(x_test, y_test), callbacks=[earlyStopping, tb_hist])
    # └ model.fit을 50번 수행하면 이전의 경험 상태를 유지해주어야한다.
    #  └ shuffle = False 를 주게 되면 그 경험상태를 유지해줄 수 있다. 
    model.reset_states()    # 상태를 리셋하지만, 이전의 경험상태를 유지한다
    histories.append(history)


print(history)

mse, _ = model.evaluate(x_test, y_test, batch_size=batch_size)
print('mse :', mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)





# RMSE 함수 적용
rmse = RMSE(y_test, y_predict)

# R2 함수 적용
y_r2 = r2_score(y_test, y_predict)

print('test :',y_test[0:10])
print('predict :',y_predict[0:10])
print('rmse :', rmse)
print('y_r2 :', y_r2)
# matplotlib 이미지 적용
legend_list = []
for i in range(len(histories)):
    plt.plot(histories[i].history['mean_squared_error'])
    legend_list.append('mse' + str(i+1))
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
#plt.legend(legend_list)
plt.show()
