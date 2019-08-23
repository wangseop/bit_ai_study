###### 데이터 ######
from keras.datasets import mnist

import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
# └비지도학습이므로 y data가 필요하지 않아요


x_train = x_train.astype('float32')/ 255
x_test = x_test.astype('float32')/ 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

################# 모델 구성 ####################
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model

# 인코딩될 표현(representation) 의 크기
encoding_dim = 32

def build_network(optimizer='adam'):
    # 입력 플레이스 홀더
    input_img = Input(shape=(784,))
    # 'encoded'는 입력의 인코딩된 표현
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded) # 784 -> 32 -> 784
    autoencoder.summary()
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return autoencoder

def create_hyperparameters():
    batches = [32,64,128,256,512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    epochs = [50,100,150,200, 250]
    return {"batch_size":batches, "optimizer":optimizers, "epochs":epochs}      # Map 형태로 반환


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

autoencoder = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()
seed = 77

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) 
# pipe = Pipeline([('minmax', MinMaxScaler()), ('model', autoencoder)])
search = RandomizedSearchCV(estimator=autoencoder, param_distributions=hyperparameters,
                            n_iter=10, n_jobs=-1, cv=kfold)

search.fit(x_train, x_train)

print(search.best_params_)

print('score :', search.score(x_test, x_test))
# 숫자들을 인코딩 / 디코딩
# test_set에서 숫자들을 가져왔다는 것을 유의
decoded_imgs = search.predict(x_test)


########################### 이미지 출력 ###############################

# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10  # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20,4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # 재구성된 데이터
    ax = plt.subplot(2, n, i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


################### 그래프 출력 ######################
def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Traning data', 'Validation data'], loc=0)
    # plt.show()

def plot_loss(history, title=None):
    # summarizer history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Traning data', 'Validation data'], loc=0)
    # plt.show()

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)

