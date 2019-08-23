import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello Yellow")

sess = tf.Session() 
# ┗Tensorflow의 Session을 이용하여 작업

print(sess.run(hello))

