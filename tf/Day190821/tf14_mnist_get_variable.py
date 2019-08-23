import tensorflow as tf
import matplotlib.pyplot as plt
import random

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


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(mnist.train.images)
# print(mnist.test.labels)
# print(mnist.train.images.shape)
# print(mnist.test.labels.shape)
# print(type(mnist.train.images))

#################################################
####  코딩하시오. X, Y, W, b, hypothesis, cost, train
#######################################################
X = tf.placeholder(tf.float32, shape=[None, 28*28])
Y = tf.placeholder(tf.float32, shape=[None, 10])

layer1, next_input_dim = create_sigmoid_layer(X, 28*28, 10, weight_name="weight1", bias_name="bias1")
layer2, next_input_dim = create_sigmoid_layer(layer1, next_input_dim, 10, weight_name="weight2", bias_name="bias2")
# layer2 = tf.nn.dropout(layer2, keep_prob=0.1)
# layer3, next_input_dim = create_sigmoid_layer(layer2, next_input_dim, 50, weight_name="weight3", bias_name="bias3")
# layer4, next_input_dim = create_sigmoid_layer(layer3, next_input_dim, 50, weight_name="weight4", bias_name="bias4")
# layer5, next_input_dim = create_sigmoid_layer(layer4, next_input_dim, 50, weight_name="weight5", bias_name="bias5")
# layer5 = tf.nn.dropout(layer5, keep_prob=0.1)
# layer6, next_input_dim = create_sigmoid_layer(layer5, next_input_dim, 100, weight_name="weight6", bias_name="bias6")
# layer7, next_input_dim = create_sigmoid_layer(layer6, next_input_dim, 100, weight_name="weight7", bias_name="bias7")
# layer8, next_input_dim = create_sigmoid_layer(layer7, next_input_dim, 100, weight_name="weight8", bias_name="bias8")
# # layer8 = tf.nn.dropout(layer8, keep_prob=0.1)
# layer9, next_input_dim = create_sigmoid_layer(layer8, next_input_dim, 165, weight_name="weight9", bias_name="bias9")
# layer10, next_input_dim = create_sigmoid_layer(layer9, next_input_dim, 165, weight_name="weight10", bias_name="bias10")
# layer11, next_input_dim = create_sigmoid_layer(layer10, next_input_dim, 165, weight_name="weight11", bias_name="bias11")
W = tf.get_variable("weight_last", shape=[next_input_dim, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]), name="bias_last")
hypothesis = tf.nn.softmax(tf.matmul(layer2, W) + b)



cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) 
train  = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
# train  = tf.train.AdamOptimizer(learning_rate=2.763e-4).minimize(cost)

# Test Model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# prarameters
num_epochs =100
batch_size =  100
num_iterations = int(mnist.train.num_examples/ batch_size)  # batch_size에 따른 반복횟수

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Traing cycle
    # w_, b_= sess.run([W, b])
    # print(w_, b_)
    
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # batch
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y:batch_ys})
            avg_cost += cost_val / num_iterations
        
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r : r+ 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r+1]}),
    )

    plt.imshow(
        mnist.test.images[r : r+1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()

'''
Accuracy:  0.9245 (tf.zeros_initializer())
'''