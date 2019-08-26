import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

tf.set_random_seed(777)     # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

def create_layer(pre_layer=None, input_dim=None, output_dim=None, weight_name="weight", bias_name="bias"):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]), name=weight_name)
    b = tf.Variable(tf.random_normal([output_dim]), name=bias_name)
    layer = tf.sigmoid(tf.matmul(pre_layer, W) + b)

    return layer, output_dim

mnist = input_data.read_data_sets("MNIST_data/")

print(mnist.train.images)
print(mnist.test.labels)
print(mnist.train.images.shape)
print(mnist.test.labels.shape)
print(type(mnist.train.images))


x_train = mnist.train.images.reshape((55000, 28*28, 1))
x_test = mnist.test.images.reshape((10000, 28*28, 1))
y_train = mnist.train.labels
y_test = mnist.test.labels
#################################################
####  코딩하시오. X, Y, W, b, hypothesis, cost, train
#######################################################
X = tf.placeholder(tf.float32, shape=[None, 28*28])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# 2. 모델 구성
model = Sequential()

model.add(LSTM(64, input_shape=(28*28,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(132, activation='relu'))
model.add(Dense(138, activation='relu'))
model.add(Dense(1))

model.summary()

# 3.훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=50, batch_size=2048, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss :', loss)
print('y_predict(x_test) :', y_predict)
