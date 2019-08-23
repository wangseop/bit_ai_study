from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler, StandardScaler

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(train_data)
'''
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
'''

# MinMaxScaler and StandardScaler
scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

from keras import models
from keras import layers

def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

seed = 77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=1, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)      # seed 고정안하면 fold시 값이 train마다 바뀔수있다
results = cross_val_score(model, train_data, train_targets, cv=kfold)       
#└ Cross Validation = '교차검증'
#└Machine Leanring 기법을 쓰게되면 속도는 빠르기 때문에 Model의 성능을 측정하여 해당 모델을 기준으로 deep learing을 구현

import numpy as np
print(results)
print(np.mean(results))

