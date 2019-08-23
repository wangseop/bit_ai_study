# 실습
# iris.npy를 가지고 텐서플로 코딩을 하시오.
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


tf.set_random_seed(777)     # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data


cancer = np.load('./data/breast_cancer.npy')

cancer_x = cancer[:, :-1]
cancer_y = cancer[:, [-1]]



x_train, x_test, y_train, y_test = train_test_split(
    cancer_x, cancer_y, test_size=0.3
)


scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(mnist.train.images)
# print(mnist.test.labels)
# print(mnist.train.images.shape)
# print(mnist.test.labels.shape)
# print(type(mnist.train.images))

#################################################
####  코딩하시오. X, Y, W, b, hypothesis, cost, train
#######################################################
X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])


L1 = tf.layers.dense(X, 15, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 30, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 1, activation=tf.nn.sigmoid)

# layer1, next_input_dim = create_sigmoid_layer(X, 4, 10, weight_name="weight1", bias_name="bias1")
# layer2, next_input_dim = create_sigmoid_layer(layer1, next_input_dim, 10, weight_name="weight2", bias_name="bias2")
# W = tf.get_variable("weight_last", shape=[next_input_dim, 3], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([3]), name="bias_last")
# hypothesis = tf.nn.softmax(tf.matmul(layer2, W) + b)



cost = -tf.reduce_mean(Y * tf.log(L3) + (1 - Y) * tf.log(1 - L3))
train  = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.cast(L3 > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype = tf.float32))


# prarameters
num_epochs = 3000
batch_size =  1
num_iterations = int(x_train.shape[0]/ batch_size)  # batch_size에 따른 반복횟수

with tf.Session() as sess:

    # sess.run(tf.global_variables_initializer())

    # for step in range(3001):
    #     cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
    #     if step % 10 == 0:
    #         print(step, cost_val)
    
    # # Predict, Accuracy
    # h, c, a = sess.run( [L3, prediction, accuracy],
    #                     feed_dict={X: x_test, Y: y_test})
    # print('◎ Accuracy:', a) # ◎ Accuracy: 0.9649123

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Traing cycle
    # w_, b_= sess.run([W, b])
    # print(w_, b_)
    
    for epoch in range(num_epochs):
        avg_cost = 0
        # for i in range(num_iterations):
        #     batch_xs = x_train[i * batch_size: (i+1) * batch_size]
        #     batch_ys = y_train[i * batch_size: (i+1) * batch_size]
        #     _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y:batch_ys})
        #     avg_cost += cost_val / num_iterations
        
        # print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X:x_train, Y: y_train})

        if epoch % 100 == 0:
            print("Step: {:5}\tCost: {:.9f}\tAcc:{:.5%}".format(epoch, cost_val, acc_val))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X:x_test, Y:y_test}
        ),
    )

    pred = sess.run(tf.argmax(L3, 1), feed_dict={X: x_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))   

'''
Accuracy: 0.98245615
'''