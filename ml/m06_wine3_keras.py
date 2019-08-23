import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout,  BatchNormalization
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

newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = np.array(newlist)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()


x_train, x_test, y_train, y_test = train_test_split(
                                        x, y, test_size=0.2                                    
)


# 학습하기
model = Sequential()
model.add(Dense(3, input_dim=11, activation='relu'))
model.add(Dense(256))
model.add(Dense(5))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(3, activation='softmax'))
# model.add(Dense(3, input_dim=11, activation='relu'))
# model.add(Dense(128))
# model.add(Dense(2))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(3, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

_, score = model.evaluate(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print('y_test :', y_test)
print('y_pred :', np.round(y_pred))
print("Score :", score)
