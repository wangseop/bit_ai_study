#Linear Regression
import tensorflow as tf
tf.set_random_seed(777) # for reproductibility

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer 
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run(
            [train, cost, W, b]
            , feed_dict={X:[1,2,3,4,5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]}
            #└  [train, cost, W, b] 찾아가는 과정은 tensor의 그래프를 가지고 가장 말단 노드에서부터 거꾸로 올라가는 방식으로 진행
        
        )
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    
    # Testing our model
    # model.predict() 하는 과정과 유사
    print(sess.run(hypothesis, feed_dict={X:[5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))