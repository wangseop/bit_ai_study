import tensorflow as tf

# tf Graph Input
X = [1,2,3]
Y = [1,2,3]

# Set wrong model weights
W = tf.Variable(5.0)        # Weight 값 지정

# Linear model
hypothesis = X * W

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, W_val = sess.run([train, W])
        print(step, W_val)      #  처음 weight 값을 임의로 주었지만, W_val은 cost 를 최소화하기 위해서 점차 1에 가깝게 나타난다