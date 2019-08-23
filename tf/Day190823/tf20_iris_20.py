# 실습
# iris.npy를 가지고 텐서플로 코딩을 하시오.
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.model_selection import train_test_split
'''
# W1 = tf.get_variable("W1", shape=[?, ?], initializer=tf.random_uniform_initializer())
# └ get_variable은 적용한 대상만 variable 초기화
# b1 = tf.variable(tf.random_normal([512]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# L1 = tf.nn.dropout(L1, leep_prob=keep_prob)

tf.constant_initializer()
tf.zeros_initializer()          ->
tf.random_uniform_initializer()
tf.random_normal_initializer()
tf.contrib.layers.xavier_initializer()      # 평균적으로 성능 우수


적용해보기
'''
nb_classes = 3
##

tf.set_random_seed(777)     # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

def create_sigmoid_layer(pre_layer=None, input_dim=None, output_dim=None, weight_name="weight", bias_name="bias"):
    # W = tf.get_variable(weight_name, shape=[input_dim, output_dim], initializer=tf.random_uniform_initializer())
    W = tf.get_variable(weight_name, shape=[input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_dim]), name=bias_name)
    layer = tf.sigmoid(tf.matmul(pre_layer, W) + b)

    return layer, output_dim

def create_relu_layer(pre_layer=None, input_dim=None, output_dim=None, weight_name="weight", bias_name="bias"):
    W = tf.get_variable(weight_name, shape=[input_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_dim]), name=bias_name)
    layer = tf.nn.relu(tf.matmul(pre_layer, W) + b)

    return layer, output_dim

iris = np.load('./data/iris.npy')

iris_x = iris[:, :-1]
iris_y = iris[:, [-1]]

x_train, x_test, y_train, y_test = train_test_split(
    iris_x, iris_y, test_size=0.2
)


# print(mnist.train.images)
# print(mnist.test.labels)
# print(mnist.train.images.shape)
# print(mnist.test.labels.shape)
# print(type(mnist.train.images))

#################################################
####  코딩하시오. X, Y, W, b, hypothesis, cost, train
#######################################################
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

L1 = tf.layers.dense(X, 10, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 10, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 10, activation=tf.nn.relu)
L_output = tf.layers.dense(L3, 3, activation=tf.nn.softmax)

# layer1, next_input_dim = create_sigmoid_layer(X, 4, 10, weight_name="weight1", bias_name="bias1")
# layer2, next_input_dim = create_sigmoid_layer(layer1, next_input_dim, 10, weight_name="weight2", bias_name="bias2")
# W = tf.get_variable("weight_last", shape=[next_input_dim, 3], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([3]), name="bias_last")
# hypothesis = tf.nn.softmax(tf.matmul(layer2, W) + b)



cost = tf.reduce_mean(-tf.reduce_sum(tf.cast(Y_one_hot, dtype=tf.float32) * tf.log(L_output), axis=1)) 
train  = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(L_output, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# prarameters
num_epochs =1000
batch_size =  32
num_iterations = int(x_train.shape[0]/ batch_size)  # batch_size에 따른 반복횟수

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Traing cycle
    # w_, b_= sess.run([W, b])
    # print(w_, b_)
    
    for epoch in range(num_epochs):
        avg_cost = 0
        for i in range(num_iterations):
            batch_xs = x_train[i * batch_size: (i+1) * batch_size]
            batch_ys = y_train[i * batch_size: (i+1) * batch_size]
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y:batch_ys})
            avg_cost += cost_val / num_iterations
        
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X:x_train, Y: y_train})

        if epoch % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc:{:.2%}".format(epoch, cost_val, acc_val))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X:x_test, Y:y_test}
        ),
    )

    pred = sess.run(tf.argmax(L_output, 1), feed_dict={X: x_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(int(p) == int(y), int(p), int(y)))   

'''
Accuracy:  0.96666664
[True] Prediction: 0 True Y: 0
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 0 True Y: 0
[True] Prediction: 2 True Y: 2
[True] Prediction: 0 True Y: 0
[True] Prediction: 2 True Y: 2
[True] Prediction: 0 True Y: 0
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 2 True Y: 2
[True] Prediction: 0 True Y: 0
[True] Prediction: 1 True Y: 1
[True] Prediction: 2 True Y: 2
[True] Prediction: 1 True Y: 1
[False] Prediction: 1 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 2 True Y: 2
[True] Prediction: 1 True Y: 1
[True] Prediction: 1 True Y: 1
[True] Prediction: 0 True Y: 0
[True] Prediction: 1 True Y: 1
'''