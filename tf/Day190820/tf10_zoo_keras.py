from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, TensorBoard




# 상수 정의
BATCH_SIZE = 2048   # 128
NB_EPOCH = 2000
NB_CLASSES = 7
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = Adam()

# 데이터셋 불러오기
xy = np.loadtxt("./data/data-04-zoo.csv", delimiter=',')



X_train = xy[:, :-1]
Y_train = xy[:, [-1]]
# 이미지 한장 보기
# digit = X_train[15]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()


# 범주형으로 변환
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES) 

# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3)


#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

# 신경망 정의
model = Sequential()
model.add(Dense(7, input_shape=(16, ), activation='softmax'))

model.summary()

# 학습
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, 
                metrics=['accuracy'])

# 텐서보드
tb_hist = TensorBoard(log_dir='./graph', 
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
# EalyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=50)


history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs= NB_EPOCH, 
                validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[earlyStopping, tb_hist])

# history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs= NB_EPOCH, 
#                 validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
# └ validation이 따로 data 없을 때 사용 할 수있다.
# └ validation_split 비율 값을 주게 되면 해당 비율 만큼 train에서 제외하고 제외한 값들은 validation data 로 사용
print("Testing...")
# score = model.evaluate(X_test, Y_test, 
#                         batch_size=BATCH_SIZE, verbose=VERBOSE)
score = model.evaluate(X_train, Y_train, 
                        batch_size=BATCH_SIZE, verbose=VERBOSE)


y_pred = np.round(model.predict(X_train))

print('\nTest score:', score[0])
print('\nTest accuracy:', score[1])
print('Y_test: ', Y_train)
print('Y_predict: ', y_pred)

# # 히스토리에 있는 모든 데이터 나열
# print(history.history.keys())
# # 단순 정확도에 대한 히스토리 요악(시각화)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train acc', 'test acc'], loc='upper left')
# plt.show()
# # └acc와 val_acc를 plot해준다

# # 손실에 대한 히스토리 요약(시각화)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train loss', 'test loss'], loc='upper left')
# plt.show()
# # └loss val_loss를 plot해준다
