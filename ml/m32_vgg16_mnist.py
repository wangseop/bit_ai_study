from keras.applications import VGG16 ,VGG19, Xception, InceptionV3, ResNet50, MobileNet
from keras.layers import Dense, Flatten
from keras.models import Model, Sequential
from keras.utils import np_utils
import numpy as np

# data
mnist_train_x = np.load("./data/mnist_train_x.npy")
mnist_test_x = np.load("./data/mnist_test_x.npy")
mnist_train_y = np.load("./data/mnist_train_y.npy")
mnist_test_y = np.load("./data/mnist_test_y.npy")

train_x = mnist_train_x.astype('float32') / 255
test_x = mnist_test_x.astype('float32') / 255

train_y = np_utils.to_categorical(mnist_train_y)
test_y = np_utils.to_categorical(mnist_test_y)

# 2차원으로 맞추기
train_x = train_x.reshape((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1]*test_x.shape[2]))

# 1채널 -> 3채널로 (3차원 형태)
train_x = np.dstack([train_x] * 3)
test_x = np.dstack([test_x] * 3)

print(train_x.shape)
print(test_x.shape)

# 4차원으로 맞추기
train_x = train_x.reshape((train_x.shape[0], 28, 28, 3))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 3))

# 48 * 48 픽셀로 맞춰주기
from keras.preprocessing.image import img_to_array, array_to_img
train_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_x])
test_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_x])

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(48,48, 3))     # VGG16은 3채널 필요, 픽셀수(48, 48 이상)도 고려해야한다
#conv_base = VGG16()   # default input_shape=(224,224,3) , default include_top=True


model = Sequential()
model.add(conv_base)
model.add(Flatten())        # conv_base가 이미 Flatten() 상태였다면 사용시 Error 생길 수 있다
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x ,train_y, epochs=10, batch_size=256, verbose=1)

print("score :", model.evaluate(test_x, test_y)[1])



