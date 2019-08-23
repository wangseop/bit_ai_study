# Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777) # for reporoducibility

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

X = tf.placeholder("float", shape=[None, 4])
Y = tf.placeholder("float", shape=[None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) +b)
# └ hypothesis에 softmax 함수 적용
#  └ nn == neural network

# cross entropy cost/ loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))       # axis=1 이므로 행단위로 합
# └ categorical crossentropy의 계산식
#   └  그래서  Keras 의 loss로 들어간다.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:x_data, Y: y_data})

        if step % 200 == 0:
            print(step, cost_val)
    
    print("-"*15)
    # Testing % One-hot encoding
    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))
                    # tf.argmax(a, 1) => a array에서 각 행에서 가장 큰 값을 가지는 인덱스번호
                    # tf.argmax(a, 1) 로 뽑히는 인덱스 값은 onehot encoding의 결과를 나타낸다
                    # tf.argmax(a, 0) => a array 에서 각 열에서 가장 큰 값을 가지는 인덱스 번호 
    
    print("-"*15)
    b = sess.run(hypothesis, feed_dict={X:[[1,3,4,3]]})
    print(b, sess.run(tf.argmax(b,1)))

    print("-"*15)
    c = sess.run(hypothesis, feed_dict={X:[[1,1,0,1]]})
    print(c, sess.run(tf.argmax(c,1)))

    print("-"*15)
    all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9], [1,3,4,3], [1,1,0,1]]})
    print(all, sess.run(tf.argmax(all,1)))

