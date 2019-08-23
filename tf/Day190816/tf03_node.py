import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
'''
┗
node 1 : tensorflow 상수 값을 실수형의 3.0 값으로 지정
node 1 : tensorflow 상수 값을 4.0 값으로 지정
node 3 : node1 + node2 를 해준다
로 나오지 않는다...
'''

print("node1 :", node1, " node2 :", node2)
print("node3 :", node3)

'''
└
node1 : Tensor("Const:0", shape=(1,), dtype=float32)  node2 : Tensor("Const_1:0", shape=(1,), dtype=float32)
node3 : Tensor("Add:0", shape=(1, 1), dtype=float32)

node명을 print하게되면 node 형태 정보를 출력해준다
tensor 값 저장 및 연산은 기계에서 이해하는 방식이므로 이를  사람이 이해하기 위해서는
Tensorflow에서 제공하는 Session을 통해 해석하여 보여줄 수 있도록 해야한다.
'''

sess = tf.Session()
print("sess.run(node1, node2) :", sess.run([node1, node2]))
print("sess.run(node3) :", sess.run(node3))