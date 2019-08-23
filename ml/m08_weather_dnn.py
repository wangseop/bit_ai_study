from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df['연'] <= 2015)
test_year = (df['연'] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []  # 학습 데이터
    y = []  # 결과
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# 직선 회귀 분석하기
lr = Sequential()
lr.add(Dense(3, input_shape=(6,), activation='relu'))
lr.add(Dense(5, activation='relu'))
lr.add(Dense(7, activation='relu'))
lr.add(Dense(12, activation='relu'))
lr.add(Dense(16, activation='relu'))
lr.add(Dense(1))

lr.compile(loss='mse', optimizer='adam', metrics=['mse'])
lr.fit(train_x, train_y, epochs=100, batch_size=None)    # 학습하기

_, mse = lr.evaluate(test_x, test_y)

pre_y = lr.predict(test_x)  # 예측하기
print('mse : ', mse)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(test_y, pre_y))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(test_y, pre_y)
print('R2 :', r2_y_predict)


# 결과를 그래프로 그리기
plt.figure(figsize=(10,6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('tenki-kion-lr.png')
plt.show()