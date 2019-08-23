# Logistic Regression Classifier
import tensorflow as tf
tf.set_random_seed(777) # for reproducibility

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) +b)
# └ 기존에 구했던 hypothesis 값(tf.matmul(X, W) + b) 을 sigmoid 함수에 Mapping하여 0과 1사이의 hypothesis 갖도록 해준다.

# cost/loss fuction 로지스틱 리그레션에서 cost에 - 가 붙는다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# └  binary crossentropy를 풀어서 쓴 식 

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accurray computation
# True if hypothesis > 0.5 else False

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# └ 회귀모델에서의 정확도를 측정하는 방식과는 맞지않는다.

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy repot
    h, c, a = sess.run([hypothesis, predicted, accuracy], 
                        feed_dict={X: x_data, Y:y_data})
    print("\nHypothesis:", h, "\nCorrect (Y):", c, "\nAccuracy:", a)
