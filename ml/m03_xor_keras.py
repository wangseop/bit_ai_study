# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = np.array([[0,0], [1, 0], [0,1],[1, 1]])
y_data = [0,1,1,0]

# 2. 모델
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# └Layer를 늘린다 -> 미분개념

# └  model 훈련 결과 => 회귀 or 분류, 
#   └ and, or는 회귀 모델로 표현해도 결과 자체는 분류의 형태를 띠게끔 나타난다. 
#    └ 소수의 data를 적합하게 만들 수 있지만, 만들게 되면 과적합이 일어난다(데이터 추가되면 모델이 안맞을수있게된다.)

# 3. 실행
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=300)

# 4. 평가 예측
x_test = np.array([[0,0], [1, 0], [0,1], [1, 1]])
_, acc = model.evaluate(x_test, y_data, batch_size=1)
y_predict = model.predict(x_test)


print(x_test, "의 예측결과 :", np.round(y_predict))
# print(x_test, "의 예측결과 :", y_predict)
print("acc =", acc)
