import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop("quality", axis=1)
print(x.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y.values.reshape(-1,1)).toarray()


x_train, x_test, y_train, y_test = train_test_split(
                                        x, y, test_size=0.2                                    
)


# 학습하기
model = Sequential()
model.add(Dense(128, input_dim=11, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=11)

_, score = model.evaluate(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print('y_test :', y_test)
print('y_pred :', np.round(y_pred))
print("Score :", score)
