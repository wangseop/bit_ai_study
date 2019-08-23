import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)     # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

def create_layer(pre_layer=None, input_dim=None, output_dim=None, weight_name="weight", bias_name="bias"):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]), name=weight_name)
    b = tf.Variable(tf.random_normal([output_dim]), name=bias_name)
    layer = tf.sigmoid(tf.matmul(pre_layer, W) + b)

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


layer1, next_input_dim = create_layer(X, 28*28, 10, weight_name="weight1", bias_name="bias1")
layer2, next_input_dim = create_layer(layer1, next_input_dim, 10, weight_name="weight2", bias_name="bias2")
layer3, next_input_dim = create_layer(layer2, next_input_dim, 15, weight_name="weight3", bias_name="bias3")
layer4, next_input_dim = create_layer(layer3, next_input_dim, 15, weight_name="weight4", bias_name="bias4")
layer5, next_input_dim = create_layer(layer4, next_input_dim, 15, weight_name="weight5", bias_name="bias5")
layer6, next_input_dim = create_layer(layer5, next_input_dim, 25, weight_name="weight6", bias_name="bias6")
layer7, next_input_dim = create_layer(layer6, next_input_dim, 25, weight_name="weight7", bias_name="bias7")
layer8, next_input_dim = create_layer(layer7, next_input_dim, 32, weight_name="weight8", bias_name="bias8")
layer9, next_input_dim = create_layer(layer8, next_input_dim, 32, weight_name="weight9", bias_name="bias9")
layer10, next_input_dim = create_layer(layer9, next_input_dim, 32, weight_name="weight10", bias_name="bias10")
layer11, next_input_dim = create_layer(layer10, next_input_dim, 32, weight_name="weight11", bias_name="bias11")

W = tf.Variable(tf.random_normal([next_input_dim, 10]), name="weight_last")
b = tf.Variable(tf.random_normal([10]), name="bias_last")
hypothesis = tf.nn.softmax(tf.matmul(layer11, W) + b)



cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) 
train  = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# train  = tf.train.AdamOptimizer(learning_rate=2.763e-4).minimize(cost)

# Test Model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# prarameters
num_epochs =15
batch_size =  100
num_iterations = int(mnist.train.num_examples/ batch_size)  # batch_size에 따른 반복횟수

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Traing cycle
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


"""

Epoch: 0001, Cost: 10.080241850
Epoch: 0002, Cost: 5.458269877
Epoch: 0003, Cost: 3.536683758
Epoch: 0004, Cost: 2.540577159
Epoch: 0005, Cost: 1.967584032
Epoch: 0006, Cost: 1.606509970
Epoch: 0007, Cost: 1.363923089
Epoch: 0008, Cost: 1.192450106
Epoch: 0009, Cost: 1.066349846
Epoch: 0010, Cost: 0.970086838
Epoch: 0011, Cost: 0.894977312
Epoch: 0012, Cost: 0.834747323
Epoch: 0013, Cost: 0.785407109
Epoch: 0014, Cost: 0.744682109
Epoch: 0015, Cost: 0.710506351
Learning finished
Accuracy:  0.8506
Label: [9]
Prediction:  [9]



"""