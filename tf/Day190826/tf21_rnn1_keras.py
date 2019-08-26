import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 데이터 구축
 
idx2char = ['e','h','i','l', 'o']

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1,1)   # (1, 7) -> (7, 1)
print(_data.shape) # (7,1)
print(_data)
print(_data.dtype)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')        
### └ OneHot Encoding 과정에서 알파벳 순서대로 정렬되어 OneHot된다(e => 1 0 0 0 0)
### └ enc.transform(_data).toarray() => float64 type으로 float32 로 형변환 해준다

print(_data)
print(_data.shape)  # (7,5) =>  oneHot 되어 column이 5개로 증가
print(type(_data))
print(_data.dtype)

x_data = _data[:6, ]    # (6,5)     hihell 부분
y_data = _data[1:, ]    # (6,5)     ihello 부분
# y_data = np.argmax(y_data, axis=1)

print(x_data)
print(y_data)

x_data = x_data.reshape(1,6,5)  # (1,6,5)
y_data = y_data.reshape(6,5)

print(x_data.shape) # (1,6,5)
print(x_data.dtype)
print(y_data.shape) # (1,6)

# 데이터 구성
# x : (batch_size, sequeqnce_length, input_dim) 1,6,5
# 첫번째 아웃풋 : hidden_size = 2
# 첫번째 결과 : 1,6,5

num_classes = 5 
batch_size = 1          # (전체행)
sequence_length = 6     # 컬럼
input_dim = 5           # 몇개씩 작업
hidden_size = 5         # 첫번째 노드 출력 개수
learning_rate = 0.1

optimizer = Adam(lr=0.01)
def rnn_not_softmax(optimizer=optimizer):
    model = Sequential()
    model.add(LSTM(64, input_shape=(6,5)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(212, activation="relu"))


    model.add(Dense(6, activation="relu"))

    model.compile(loss="mse", optimizer=optimizer, metrics=["acc"])
    return model

def rnn_softmax(optimizer=optimizer):

    model = Sequential()
    model.add(LSTM(64, input_shape=(6,5)))
    model.add(Dense(128, activation="relu"))


    model.add(Dense(5, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    return model


model = rnn_softmax(optimizer=optimizer)
# model = rnn_not_softmax(optimizer=optimizer)

from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='acc', patience=30, mode='auto')
model.fit(x_data, y_data, epochs=500, batch_size=4, verbose=1
                        # ,callbacks=[early_stopping]
                        )

_, acc = model.evaluate(x_data, y_data)

y_predict = model.predict(x_data)

print('acc :', acc)
print('y_predict(x_test) :', np.round(y_predict).astype('int32'))
y_pred_idx = np.round(y_predict).astype('int32')

result_str = [idx2char[c] for c in np.squeeze(y_pred_idx)]      

print("\nprediction str :", ''.join(result_str))
