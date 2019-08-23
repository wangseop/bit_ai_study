# Minimizing cost
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)

# Out hypothesis for linear model X * W
hypothesis = X * W

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Variables for plotting cost function
W_history = []
cost_history = []

# Launch the graph in a session.
with tf.Session() as sess:
    for i in range(-30, 50):
        curr_W = i * 0.1            # feed 로 줄 weight 값
        curr_cost = sess.run(cost, feed_dict={W:curr_W})    # weight값에 따른 cost

        # 그래프에 출력할 내용으로 추가
        W_history.append(curr_W)
        cost_history.append(curr_cost)

# show the cost function
plt.plot(W_history, cost_history)       
plt.show()