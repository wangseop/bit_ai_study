# keras32_hyperParameter.py -> gridSearchCV 로 변경

# 데이터 불러오기

from keras.datasets import mnist
from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
'''
import matplotlib.pyplot as plt

digit = X_train[11]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''
# └ 눈으로 보여주게 한다 -> 데이터 시각화
# └ plt.show() 는 jupyter notebook 에서 사용시엔 안써도 시각화된다. python에서는 써줘야한다

X_train = X_train.reshape(X_train.shape[0], 28 * 28 * 1).astype('float32')/255   
X_test = X_test.reshape(X_test.shape[0], 28 * 28 * 1).astype('float32')/255       # 60000, 28, 28 =?> 60000 28 28 1
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)      # 분류값 집어넣는다 (0 ~ 9), onehot encoding 방식 사용
Y_test = np_utils.to_categorical(Y_test)        # 0000001000 : 6 값을 의미
print(Y_train.shape)
print(Y_test.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):

    
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(10, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(10, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(10, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5, 5)
    return [{"batch_size":batches, "optimizer":[optimizers[0]], "keep_prob":dropout}, 
            {"batch_size":batches, "optimizer":[optimizers[1]], "keep_prob":dropout}, 
            {"batch_size":batches, "optimizer":[optimizers[2]], "keep_prob":dropout}]      # Map 형태로 반환

from keras.wrappers.scikit_learn import KerasClassifier # 사이킷런과 호환하도록 함.
# from keras.wrappers.scikit_learn import KerasRegressor # 사이킷런과 호환하도록 함.
# └cross validation : 교차로 데이터를 교체하면서 검증, Scikitlearn에서는 제공하지만 Keras 에서는 제공하지않는다
#  └ 교차검증을 적용하기 위해서 Wrapper로 감싸서 Keras를 scikitlearn에서 돌아가도록 해준다 

model = KerasClassifier(build_fn=build_network, verbose=1)    # verbose = 0

hyperparameters = create_hyperparameters()

# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=3, verbose=1)

# search.fit(data["X_train"], data["Y_train"])
search.fit(X_train, Y_train)

print(search.best_params_)
# └ create_hyperparameters()에서 리턴값으로 가지는 것 들 중 각각 최적의 parameter를 뽑아준다
#   └ 현재 모델 기준으로 가장 좋은 최적의 parameter이다 
