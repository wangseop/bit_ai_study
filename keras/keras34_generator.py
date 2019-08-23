from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)



# # seed 값 설정
# seed = 0
# numpy.random.seed(seed)
# tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# └ Data 양이 많고 전처리가 잘 되어있는 데이터 => 정확도가 높다
# └ 만일 data양을 줄인다면? 정확도가 떨어질 것이다
'''
import matplotlib.pyplot as plt
digit = X_train[11]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''
X_train = X_train[:300]
Y_train = Y_train[:300]
# X_test = X_test[:300]
# Y_test = Y_test[:300]
# └ 눈으로 보여주게 한다 -> 데이터 시각화
# └ plt.show() 는 jupyter notebook 에서 사용시엔 안써도 시각화된다. python에서는 써줘야한다

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255   
     
#데이터 전처리 => 0~1 사이 값으로(minmaxscaler와 유사)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255       # 60000, 28, 28 =?> 60000 28 28 1
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train)      # 분류값 집어넣는다 (0 ~ 9), onehot encoding 방식 사용
Y_test = np_utils.to_categorical(Y_test)        # 0000001000 : 6 값을 의미
print(Y_train.shape)
print(Y_test.shape)


# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(30, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(40, (3,3), activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))      # 분류모델 마지막은 softmax로 

model.compile(loss='categorical_crossentropy',  # 분류 모델은 loss로 categorical_crossentropy 사용
            optimizer='adam',
            metrics=['accuracy'])
model.summary()

# 모델 최적화 설정

early_stopping_callback = EarlyStopping(monitor='loss', patience=20)

# 모델의 실행


model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=256),
                    steps_per_epoch=len(X_train),     # 증폭시킬 양
                    epochs=20,
                    validation_data=(X_test, Y_test),
                    verbose=1, callbacks=[early_stopping_callback]   #, callbacks=callbacks
)
# ┗ fit & generator 동작
# ┗ 실행하면서 이미지 프로세싱
#  ┗이미지 데이터 개수가 증폭되며 그 수는 steps_per_epoch * batch_size 만큼 증폭
# 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))

# 실습 1. keras25.py(데이터가 300개인 mnist를 합체할것)
# 실습 2. acc = 99% 이상 올릴 것