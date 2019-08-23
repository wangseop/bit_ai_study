import tensorflow as tf
import numpy as np

def create_layer(pre_layer=None, input_dim=None, output_dim=None, weight_name="weight", bias_name="bias"):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]), name=weight_name)
    b = tf.Variable(tf.random_normal([output_dim]), name=bias_name)
    layer = tf.sigmoid(tf.matmul(pre_layer, W) + b)

    return layer, output_dim

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# X, Y, W, b, hypothesis, cost, train

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#############################
##### DNN   만들기 ##########
############################

'''
Deep Layer 구성

W1       -> (input1, output1)
b1       -> (output1)
Layer1   -> activation

W2       -> (input2 = output1, output2)
b2       -> (output2)
Layer2   -> activation

.
.
.

'''


W1 = tf.Variable(tf.random_normal([2, 20]), name='weight1')
b1 = tf.Variable(tf.random_normal([20]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# └ Layer를 늘려 xor문제를 해결, 위처럼 구성하면 1개의 layer가 생성된다

W2 = tf.Variable(tf.random_normal([20, 30]), name='weight2')
b2 = tf.Variable(tf.random_normal([30]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([30, 40]), name='weight3')
b3 = tf.Variable(tf.random_normal([40]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([40, 50]), name='weight4')
b4 = tf.Variable(tf.random_normal([50]), name='bias4')
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([50, 60]), name='weight5')
b5 = tf.Variable(tf.random_normal([60]), name='bias5')
layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([60, 70]), name='weight6')
b6 = tf.Variable(tf.random_normal([70]), name='bias6')
layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_normal([70, 80]), name='weight7')
b7 = tf.Variable(tf.random_normal([80]), name='bias7')
layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_normal([80, 90]), name='weight8')
b8 = tf.Variable(tf.random_normal([90]), name='bias8')
layer8 = tf.matmul(layer7, W8) + b8

W9 = tf.Variable(tf.random_normal([90, 100]), name='weight9')
b9 = tf.Variable(tf.random_normal([100]), name='bias9')
layer9 = tf.sigmoid(tf.matmul(layer8, W9) + b9)

layer_func = create_layer(pre_layer=layer9, input_dim=100, output_dim=110, weight_name="weight_func", bias_name="bias_func")

W10 = tf.Variable(tf.random_normal([110, 1]), name='weight10')
b10 = tf.Variable(tf.random_normal([1]), name='bias10')
hypothesis = tf.sigmoid(tf.matmul(layer_func, W10) + b10)


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# sigmoid
# predict, accuracy

# Launcy graph
with tf.Session() as sess:
    # Initialize Tensorflow variables 
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run(
            [train, cost, W2], feed_dict={X: x_data, Y:y_data}
        )
        if step % 100 == 0:
            print(step, cost_val, w_val)
    # Accuracy report
    h, c, a = sess.run(
        [hypothesis, predicted, accuracy], feed_dict={X: x_data, Y:y_data}
    )
    print("\nHypothesis: ", np.round(h), "\nCorrect: ", c, "\nAccuracy:", a)


