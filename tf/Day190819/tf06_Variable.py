#  랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력하시오.

import tensorflow as tf
tf.set_random_seed(777) # for reproducibility


W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

print(W)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # └  안해주면  variable의 값이 출력되지 않는다.
    #   └  기존의 강한  type의 언어에서 변수 값을 초기화해주는 것과 같은 개념으로 이해하면 된다.
    #       └ 안했을 경우 오류가 발생하고, 이는 쓰레기값을 가지고 출력하는 것과 비슷하다.
    print(sess.run(W))
    print(sess.run(b))



#######  Variable 의 다양한 표현들 ######

W = tf.Variable([0.3], tf.float32)          #  W = 0.3 처럼 이해한다.

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     print(sess.run(W))


# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = W.eval()      #  기존 Session에서 사용햇던 sess.run(W) 대신 사용
# print(aaa)
# sess.close()

# eval에서 Session을 명시 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = W.eval(session=sess)        
print(aaa)
sess.close()