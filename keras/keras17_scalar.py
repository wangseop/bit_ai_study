import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1,11))
size = 5

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 5개씩 자르는 함수
def split_5(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
dataset = split_5(a, size)
x_train = dataset[:,0:4]
y_train = dataset[:,4]
print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)
print("========================")   
'''
#scalar = MinMaxScaler()
scalar = StandardScaler()
scalar.fit(x_train)
x_train_scaled = scalar.transform(x_train)
print(x_train)
# x_train = np.reshape(x_train, (6, 4, 1))
# x_train = np.reshape(x_train, (len(a)-size+1, 4, 1))
# x_train_scaled = np.reshape(x_train_scaled, (len(a)-size+1, 4, 1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(x_train.shape)


#x_test = np.array([[11,12,13,14], [12,13,14,15], [13,14,15,16], [14,15,16,17]])
x_test = np.array([[[11],[12],[13],[14]], [[12],[13],[14],[15]], [[13],[14],[15],[16]], [[14],[15],[16],[17]]])
y_test = np.array([15,16,17,18])

x_test_scaled = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
x_test_scaled = scalar.transform(x_test_scaled)
x_test_scaled = np.reshape(x_test_scaled, (x_test.shape[0], x_test.shape[1], 1))


print(x_test.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()

model.add(LSTM(32, input_shape=(4,1)))
model.add(Dense(500, activation='relu'))

model.add(Dense(160, activation='relu'))
model.add(Dense(140, activation='relu'))
model.add(Dense(140, activation='relu'))
model.add(Dense(140, activation='relu'))

model.add(Dense(140, activation='relu'))
model.add(Dense(140, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.summary()

# 3.훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='auto')
model.fit(x_train_scaled, y_train, epochs=10000, batch_size=5, verbose=1, callbacks=[early_stopping])
# model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0, callbacks=[early_stopping])

# loss, acc = model.evaluate(x_test, y_test)
loss, acc = model.evaluate(x_test_scaled, y_test)

# y_predict = model.predict(x_test)
y_predict = model.predict(x_test_scaled)

print('loss :', loss)
print('y_predict(x_test) :', y_predict)
'''