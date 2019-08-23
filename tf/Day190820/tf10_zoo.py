# Softmax Classifier
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
tf.set_random_seed(777) # for reporoducibility

xy = np.loadtxt("./data/data-04-zoo.csv", delimiter=',')

print(xy.shape)

x = xy[:, :-1]
y = xy[:, [-1]]

print(x.shape)
print(y.shape)


X = tf.placeholder("float", shape=[None, 16])
Y = tf.placeholder("float", shape=[None, 7])

y = np_utils.to_categorical(y)

W = tf.Variable(tf.random_normal([16,7]), name='weight')
b = tf.Variable(tf.random_normal([7]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) +b)
# └ hypothesis에 softmax 함수적용
#  └ nn == neural network

# cross entropy cost/ loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))       # axis=1 이므로 행단위로 합
# └ categorical crossentropy의 계산식
#   └  그래서  Keras 의 loss로 들어간다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

'''
predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(Y, 1)), dtype=tf.float32))
'''

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:x, Y: y})

        if step % 200 == 0:
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, prediction, accuracy],
                        feed_dict={X:x , Y:y})
    print("\nHypothesis:", np.round(h), "\nCorrect (Y):", c, "\nAccuracy:", a)