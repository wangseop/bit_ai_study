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
learning_rate = 0.001
training_epochs = 100
batch_size = 1024

# input place holders
X = tf.placeholder(tf.float32, [None, IMG_ROWS,IMG_COLS,3])
Y = tf.placeholder(tf.int32, [None, 1])

# Y one hot encoding
Y_one_hot = tf.one_hot(Y, NB_CLASSES)
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NB_CLASSES])
print("reshape one_hot:", Y_one_hot)

# model Layer
L1 = tf.layers.conv2d(X, 64, [5,5],activation=tf.nn.relu, padding="SAME")
L1 = tf.layers.max_pooling2d(L1, [3,3], [2,2], padding="SAME")

L2 = tf.layers.conv2d(L1, 64, [5,5],activation=tf.nn.relu, padding="SAME")
L2 = tf.layers.max_pooling2d(L1, [3,3], [2,2], padding="SAME")

L3 = tf.layers.conv2d(L2, 128, [3,3],activation=tf.nn.relu, padding="SAME")

L4 = tf.layers.conv2d(L3, 128, [3,3],activation=tf.nn.relu, padding="SAME")

L5 = tf.layers.conv2d(L4, 128, [3,3],activation=tf.nn.relu, padding="SAME")

L6 = tf.contrib.layers.flatten(L5)

L7 = tf.layers.dense(L6, 384, activation=tf.nn.relu)
L8 = tf.layers.dense(L6, 256, activation=tf.nn.relu)
L8 = tf.layers.dense(L6, 128, activation=tf.nn.relu)

L_output = tf.layers.dense(L8, 10, activation=None)
# Dense

# define cost/ loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L_output, labels=Y_one_hot))

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
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            feed_dict = {X:batch_xs, Y:batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', "%04d" % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print('Learning finished!')

    # Test model and check accuracy

    correct_prediction = tf.equal(tf.argmax(L_output, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X:x_test, Y:y_test
    }))

    # Get one and predict
    r = random.randint(0, len(y_test) - 1)
    print("Label: ", sess.run(tf.argmax(y_test[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(L_output, 1), feed_dict={X: x_test[r:r+1]}))

    '''
    Epoch: 0001 cost= 5.490493543
Epoch: 0002 cost= 1.514811240
Epoch: 0003 cost= 1.319010456
Epoch: 0004 cost= 1.171975009
Epoch: 0005 cost= 1.061978992
Epoch: 0006 cost= 0.970748624
Epoch: 0007 cost= 0.901542320
Epoch: 0008 cost= 0.826631277
Epoch: 0009 cost= 0.760439182
Epoch: 0010 cost= 0.689326635
Epoch: 0011 cost= 0.631450785
Epoch: 0012 cost= 0.583546706
Epoch: 0013 cost= 0.520769044
Epoch: 0014 cost= 0.484443889
Epoch: 0015 cost= 0.418314965
Epoch: 0016 cost= 0.353617570
Epoch: 0017 cost= 0.333017209
Epoch: 0018 cost= 0.279409547
Epoch: 0019 cost= 0.247046387
Epoch: 0020 cost= 0.215425833
Epoch: 0021 cost= 0.203723175
Epoch: 0022 cost= 0.170287369
Epoch: 0023 cost= 0.144939079
Epoch: 0024 cost= 0.104508416
Epoch: 0025 cost= 0.111789792
Epoch: 0026 cost= 0.090993556
Epoch: 0027 cost= 0.093995744
Epoch: 0028 cost= 0.079325664
Epoch: 0029 cost= 0.071205232
Epoch: 0030 cost= 0.055496791
Epoch: 0031 cost= 0.060545402
Epoch: 0032 cost= 0.059749233
Epoch: 0033 cost= 0.056074606
Epoch: 0034 cost= 0.041488230
Epoch: 0035 cost= 0.036449295
Epoch: 0036 cost= 0.057357379
Epoch: 0037 cost= 0.085136812
Epoch: 0038 cost= 0.109063705
Epoch: 0039 cost= 0.101820589
Epoch: 0040 cost= 0.074256885
Epoch: 0041 cost= 0.068628453
Epoch: 0042 cost= 0.054376554
Epoch: 0043 cost= 0.053545562
Epoch: 0044 cost= 0.041405406
Epoch: 0045 cost= 0.037098058
Epoch: 0046 cost= 0.035009177
Epoch: 0047 cost= 0.031419423
Epoch: 0048 cost= 0.040624383
Epoch: 0049 cost= 0.060396829
Epoch: 0050 cost= 0.055764928
Epoch: 0051 cost= 0.083999596
Epoch: 0052 cost= 0.077339394
Epoch: 0053 cost= 0.093698005
Epoch: 0054 cost= 0.064395401
Epoch: 0055 cost= 0.051506075
Epoch: 0057 cost= 0.036087673
Epoch: 0058 cost= 0.042611772
Epoch: 0059 cost= 0.042186942
Epoch: 0060 cost= 0.052954641
Epoch: 0061 cost= 0.060178021
Epoch: 0062 cost= 0.044111211
Epoch: 0063 cost= 0.070057837
Epoch: 0064 cost= 0.058557971
Epoch: 0065 cost= 0.066369774
Epoch: 0066 cost= 0.052157399
Epoch: 0067 cost= 0.052964643
Epoch: 0068 cost= 0.045308832
Epoch: 0069 cost= 0.052146712
Epoch: 0070 cost= 0.045772308
Epoch: 0071 cost= 0.060677139
Epoch: 0072 cost= 0.056473463
Epoch: 0073 cost= 0.045278543
Epoch: 0074 cost= 0.044979769
Epoch: 0075 cost= 0.070548648
Epoch: 0076 cost= 0.078107717
Epoch: 0077 cost= 0.069059873
Epoch: 0078 cost= 0.061726321
Epoch: 0079 cost= 0.033301393
Epoch: 0080 cost= 0.034421495
Epoch: 0081 cost= 0.049324203
Epoch: 0082 cost= 0.067741794
Epoch: 0083 cost= 0.046363517
Epoch: 0084 cost= 0.029456713
Epoch: 0085 cost= 0.043621330
Epoch: 0086 cost= 0.071524675
Epoch: 0087 cost= 0.046579644
Epoch: 0088 cost= 0.043265699
Epoch: 0089 cost= 0.064823135
Epoch: 0090 cost= 0.086986921
Epoch: 0091 cost= 0.045261560
Epoch: 0092 cost= 0.026561384
Epoch: 0093 cost= 0.033016570
Epoch: 0094 cost= 0.035752976
Epoch: 0095 cost= 0.044621719
Epoch: 0096 cost= 0.048800728
Epoch: 0097 cost= 0.053122371
Epoch: 0098 cost= 0.045993720
Epoch: 0099 cost= 0.048023959
Epoch: 0100 cost= 0.068577819
Learning finished!
Accuracy: 0.6719
    '''