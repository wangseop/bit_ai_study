#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

# # seed 값 설정
# seed = 0
# numpy.random.seed(seed)
# tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
'''
import matplotlib.pyplot as plt

digit = X_train[11]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
'''
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
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))      # 분류모델 마지막은 softmax로 

model.compile(loss='categorical_crossentropy',  # 분류 모델은 loss로 categorical_crossentropy 사용
            optimizer='rmsprop',
            metrics=['accuracy'])
model.summary()

# 모델 최적화 설정

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, 
            batch_size=20, verbose=1, callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))
