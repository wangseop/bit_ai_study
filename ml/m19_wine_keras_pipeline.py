import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout,  BatchNormalization
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop("quality", axis=1)
print(x.shape)

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
                                        x, y, test_size=0.2, train_size=0.8                                    
)



def build_model(keep_prob=0.5, optimizer='adam'):
    # 학습하기
    model = Sequential()
    model.add(Dense(3, input_dim=11, activation='relu'))
    model.add(Dense(256))
    model.add(Dense(5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.05,0.5, 10)
    epochs = [50,100,150,200, 250]
    return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}      # Map 형태로 반환

model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

pipe = Pipeline([('minmax', MinMaxScaler()), ('model', model)])
search = RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters, n_iter=10, n_jobs=-1, cv=5)


search.fit(x_train, y_train)


# 평가하기

print("Score :", search.score(x_test, y_test))

''' 
model = Sequential()
model.add(Dense(3, input_dim=11, activation='relu'))
model.add(Dense(256))
model.add(Dense(5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(7, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

batches = [10,20,30,40,50]
optimizers = ['rmsprop', 'adam', 'adadelta']
dropout = np.linspace(0.05,0.5, 10)
epochs = [50,100,150,200, 250]
return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}      # Map 형태로 반환

RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters, n_iter=10, n_jobs=-1, cv=5)

Score : 0.9214285624270536
'''