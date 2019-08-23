# pima-indians-diabetes.csv를 파이프라인처리하시오.
# 최적의 파리미터를 구한뒤 모델링해서
# acc 확인.

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Dropout
import numpy as np
import tensorflow as tf

#  seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
# dataset = numpy.loadtxt("D:\\study\\data\\pima-indians-diabetes.csv", delimiter=",")
x = dataset[:, 0:8]
# ‪X = dataset[:, 0:8]
y = dataset[:, 8]

# 모델의 설정
def build_network(keep_prob=0.1 , optimizer='adam'):
    # model = Sequential()
    # model.add(Dense(15, input_dim=8, activation='relu'))
    # model.add(Dense(17, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(keep_prob))
    # model.add(Dense(18, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # # output에 sigmoid를 주게 되면 결과치를 0이나 1로 근사하게 도출한다
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   
    inputs = Input(shape=(8, ), name='input')
    x = Dense(50, activation='relu', name='hidden1')(inputs)
    x = Dense(512, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden6')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(512, activation='relu', name='hidden7')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.05,0.5, 10)
    epochs = [50,100,150,200, 250]
    return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}      # Map 형태로 반환

#  모델 컴파일
# sigmoid 결과로부터 0이나 1로 분류하기 위해서 loss를 binary_crossentropy 사용

hyperparameters = create_hyperparameters()

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=.8
)

model = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([('minmax', MinMaxScaler()), ('model', model)])
search = RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters, n_iter=20, n_jobs=16, cv=5)

search.fit(x_train, y_train)
print(search.best_params_)
print('score :', search.score(x_test, y_test))
# 모델 실행

# 결과 출력
# print('\n Accuracy: %.4f'%(model.evaluate(x, y)[1]))


'''
inputs = Input(shape=(8, ), name='input')
x = Dense(512, activation='relu', name='hidden1')(inputs)
x = Dropout(keep_prob)(x)
x = Dense(256, activation='relu', name='hidden2')(x)
x = Dropout(keep_prob)(x)
x = Dense(128, activation='relu', name='hidden3')(x)
x = Dropout(keep_prob)(x)
x = Dense(128, activation='relu', name='hidden4')(x)
x = Dropout(keep_prob)(x)
x = Dense(256, activation='relu', name='hidden5')(x)
x = Dropout(keep_prob)(x)
x = Dense(512, activation='relu', name='hidden6')(x)
x = Dropout(keep_prob)(x)
prediction = Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=inputs, outputs=prediction)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

batches = [10,20,30,40,50]
optimizers = ['rmsprop', 'adam', 'adadelta']
dropout = np.linspace(0.05,0.5, 10)
epochs = [50,100,150,200, 250]
return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}  

{'model__optimizer': 'adam', 'model__keep_prob': 0.2, 'model__epochs': 50, 'model__batch_size': 30}
154/154 [==============================] - 0s 497us/step
score : 0.7922077867891881
'''

'''
inputs = Input(shape=(8, ), name='input')
x = Dense(50, activation='relu', name='hidden1')(inputs)
x = Dense(512, activation='relu', name='hidden2')(x)
x = Dropout(keep_prob)(x)
x = Dense(256, activation='relu', name='hidden3')(x)
x = Dropout(keep_prob)(x)
x = Dense(128, activation='relu', name='hidden4')(x)
x = Dropout(keep_prob)(x)
x = Dense(128, activation='relu', name='hidden5')(x)
x = Dropout(keep_prob)(x)
x = Dense(256, activation='relu', name='hidden6')(x)
x = Dropout(keep_prob)(x)
x = Dense(512, activation='relu', name='hidden7')(x)
x = Dropout(keep_prob)(x)
prediction = Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=inputs, outputs=prediction)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

batches = [10,20,30,40,50]
optimizers = ['rmsprop', 'adam', 'adadelta']
dropout = np.linspace(0.05,0.5, 10)
epochs = [50,100,150,200, 250]
return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}      # Map 형태로 반환

RandomizedSearchCV(estimator=pipe, param_distributions=hyperparameters, n_iter=20, n_jobs=16, cv=5)

{'model__optimizer': 'rmsprop', 'model__keep_prob': 0.45, 'model__epochs': 100, 'model__batch_size': 50}
score : 0.8181817980555744
'''