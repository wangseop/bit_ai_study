from keras.datasets import cifar10
import tensorflow as tf
import random
import numpy as np

# Constant
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
NB_CLASSES = 10

def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


tf.set_random_seed(777)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)    # (50000, 32, 32, 3)
print(y_train.shape)    # (50000, 1)
print(x_test.shape)     # (10000, 32, 32, 3)
print(y_test.shape)     # (10000, 1)

# hyper parameters
learning_rate = 0.002
training_epochs = 50
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, IMG_ROWS,IMG_COLS,IMG_CHANNELS])
Y = tf.placeholder(tf.int32, [None, 1])

# Y one hot encoding
Y_one_hot = tf.one_hot(Y, NB_CLASSES)
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NB_CLASSES])
print("reshape one_hot:", Y_one_hot)

# model Layer
W1 = tf.Variable(tf.random_normal([5,5,3,64], stddev=5e-2))
print("W1:", W1)
L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
print("L1:", L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
print("max_pooled_L1:", L1)

W2 = tf.Variable(tf.random_normal([5,5,64,64], stddev=5e-2))
print("W2:", W2)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
print("L2:", L2)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
print("max_pooled_L2:", L2)

W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=5e-2))
print("W3:", W3)
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
print("L3:", L3)
L3 = tf.nn.relu(L3)

W4 = tf.Variable(tf.random_normal([3,3,128,128], stddev=5e-2))
print("W4:", W4)
L4 = tf.nn.conv2d(L3, W4, strides=[1,1,1,1], padding='SAME')
print("L4:", L4)
L4 = tf.nn.relu(L4)

W5 = tf.Variable(tf.random_normal([3,3,128,128], stddev=5e-2))
print("W5:", W5)
L5 = tf.nn.conv2d(L4, W5, strides=[1,1,1,1], padding='SAME')
print("L5:", L5)
L5 = tf.nn.relu(L5)

L5_flat = tf.reshape(L5, [-1,8*8*128])
print("L5 flat:", L5_flat)

# Dense
W6 = tf.Variable(tf.random_normal([8*8*128, 384]))
b6 = tf.Variable(tf.random_normal([384]))
L6 = tf.nn.relu(tf.matmul(L5_flat, W6) + b6)

W7 = tf.get_variable("W7", shape=[384, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L6, W7) + b

# define cost/ loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#initializer
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        # _, cost_val = sess.run([optimizer, cost], feed_dict={X:x_train, Y: y_train})

        for i in range(total_batch):
            # batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            
            feed_dict = {X:batch_xs, Y:batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', "%04d" % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print('Learning finished!')

    # Test model and check accuracy

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X:x_test, Y:y_test
    }))

    # Get one and predict
    r = random.randint(0, len(y_test) - 1)
    print("Label: ", sess.run(tf.argmax(y_test[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: x_test[r:r+1]}))