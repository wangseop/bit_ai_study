# Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777) # for reporoducibility

#### Constant Value ####
nb_classes = 7

########################
xy = np.loadtxt("./data/data-04-zoo.csv", delimiter=',')

print(xy.shape)

x = xy[:, :-1]
y = xy[:, [-1]]

print(x.shape)
print(y.shape)


X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

W = tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
# └ hypothesis에 softmax 함수적용
#  └ nn == neural network

# cross entropy cost/ loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient([Y_one_hot])))       # axis=1 이므로 행단위로 합
# └ tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))  를 함수화한것

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# └ tf.equal의 결과는 dtype이 bool이므로 type casting을 통해 계산할 수 있는 type인 float32로 변경 해줘야한다



# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X:x, Y: y})

        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc:{:.2%}".format(step, cost_val, acc_val))
    
    pred = sess.run(prediction, feed_dict={X: x})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))   
