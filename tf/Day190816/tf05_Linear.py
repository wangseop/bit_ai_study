import tensorflow as tf
tf.set_random_seed(777)
# ┗ seed 값으로 777 사용, 난수 표에서 777에 매핑된 내용으로 값이 랜덤하게 저장된다.

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# ┗ 

hypothesis = x_train * W + b
# ┗ 최적의 모델을 구성하기 위해서 W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))      
# ┗loss를 mse로 사용했다는 얘기

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

'''
┗위에서 cost, train을 정하는 과정 -> keras의 compile 과정
'''
# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    # ┗ 이 과정은 기계가 이해할 수 있도록 변수 값을 초기화하는 과정이다
    #  ┗ 명시적으로 써준다 생각하면 된다.

    # Fit the line
    for step in range(2001):            # keras의 epochs
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])   # keras의 fit(우측 list에서 위치한 순서대로 좌측 값에 fit하여 나온 최적의 모델 값들이 출력)

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

# with 문법은 Session의 close를 명시하지 않아도 인터프리터가 해석하는 과정에서 with 마지막 부분에 Session을 close 하는 내용을 기입하여 주어
# 사람이 직접 coding 하지 않도록 해준다.(Session 사용 시 발생할 수 있는 예외 사항을 방지해주는 역할)
