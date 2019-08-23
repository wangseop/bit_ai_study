from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # 분류

y = cancer['target']
x = cancer['data']

print(x.shape)  # (569, 30)
print(y.shape)  # (569, )

from keras.utils import np_utils
# y = np_utils.to_categorical(y)

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input

import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):

    inputs = Input(shape=(30, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5, 5)
    return {"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}      # Map 형태로 반환

from keras.wrappers.scikit_learn import KerasClassifier


from sklearn.model_selection import train_test_split, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8
)

model = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters,
                            n_iter=10, n_jobs=4, cv=3)

search.fit(x_train, y_train, epochs=300)
print(search.best_params_)
print('score : ', search.score(x_test, y_test))

# y_pred = search.predict(x_test)