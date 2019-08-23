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
    model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Dense(170, activation='relu'))
    model.add(layers.Dense(190, activation='relu'))
    model.add(layers.Dense(210, activation='relu'))
    model.add(layers.Dense(210, activation='relu'))
    model.add(layers.Dense(210, activation='relu'))
    model.add(layers.Dense(210, activation='relu'))
    model.add(layers.Dense(210, activation='relu'))
    model.add(layers.Dense(210, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

from sklearn.model_selection import KFold
import numpy as np
k = 5           # 분할 개수 ()
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores= []

# 사이킷런 KFold로 리파인
kfold = KFold(n_splits=k)
# for i in range(k):
#     print('처리중인 폴드 #', i)
#     # 검증 데이터 준비 : k번째 분할
#     val_data = train_data[i*num_val_samples: (i+1) * num_val_samples]
#     val_targets = train_targets[i* num_val_samples:(i+1)*num_val_samples]

#     # 훈련 데이터 준비
#     partial_train_data = np.concatenate(
#             [train_data[:i * num_val_samples], 
#             train_data[(i+1) *num_val_samples:]],
#             axis = 0
#     )
#     partial_train_targets = np.concatenate(
#             [train_targets[:i*num_val_samples], train_targets[(i+1) * num_val_samples:]], axis=0
#     )
 
#     # 케라스 모델 구성(컴파일 포함)
#     model = build_model()

#     # 모델 훈련(verbose=0이므로 훈련과정이 출력되지 않습니다)
#     model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=1)

#     # 검증 세트로 모델 평가
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)


for idx, (train_idx, val_idx) in enumerate(kfold.split(train_data, train_targets)):

    partial_train_data, val_data = train_data[train_idx], train_data[val_idx]
    partial_train_targets, val_targets = train_targets[train_idx], train_targets[val_idx]
    # 케라스 모델 구성(컴파일 포함)
    model = build_model()

    # 모델 훈련(verbose=0이므로 훈련과정이 출력되지 않습니다)
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=1)

    # 검증 세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)



print(all_scores)
print(np.mean(all_scores))

