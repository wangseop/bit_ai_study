from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping, TensorBoard

# CIFAR_10 은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수 정의
BATCH_SIZE = 2048   # 128
NB_EPOCH = 100
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 이미지 한장 보기
# digit = X_train[15]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()


# 범주형으로 변환
Y_train = np_utils.to_categorical(y_train, NB_CLASSES) 
# └숫자를 onehot Encoding으로
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 실수형으로 지정하고 정규화
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255             
# X_test /= 255
# └ float32로 바꾸는 것은 0~1 사이의 소수점으로 매핑하기 위해서
# └ 255 - 0 으로 나누어 minmax 적용하는 것이 일반적, but 주어진 set이 쏠린 데이터라면 minmax보단 standard가 더 나을 수 있다.

# MinMax Scaler, StandardScaler 적용
X_train0 = X_train.shape[0]
X_test0 = X_test.shape[0]
pixel_width = X_train.shape[1]
pixel_height = X_train.shape[2]
image_color = X_train.shape[3]

X_train = X_train.reshape((X_train0,  pixel_width * pixel_height * image_color))
X_test = X_test.reshape((X_test0, pixel_width * pixel_height * image_color))

#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

# 신경망 정의
model = Sequential()
model.add(Dense(256, input_shape=(32*32*3, )))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(NB_CLASSES))
model.add(Activation('relu'))
model.add(Activation('softmax'))

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
score = model.evaluate(X_test, Y_test, 
                        batch_size=BATCH_SIZE, verbose=VERBOSE)
print('\nTest score:', score[0])
print('\nTest accuracy:', score[1])

# 히스토리에 있는 모든 데이터 나열
print(history.history.keys())
# 단순 정확도에 대한 히스토리 요악(시각화)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train acc', 'test acc'], loc='upper left')
plt.show()
# └acc와 val_acc를 plot해준다

# 손실에 대한 히스토리 요약(시각화)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss'], loc='upper left')
plt.show()
# └loss val_loss를 plot해준다
