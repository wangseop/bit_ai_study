import tensorflow as tf
import numpy as np

# 데이터 구축
 
idx2char = ['e','h','i','l', 'o']

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1,1)   # (1, 7) -> (7, 1)
print(_data.shape) # (7,1)
print(_data)
print(_data.dtype)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')        
### └ OneHot Encoding 과정에서 알파벳 순서대로 정렬되어 OneHot된다(e => 1 0 0 0 0)
### └ enc.transform(_data).toarray() => float64 type으로 float32 로 형변환 해준다

print(_data)
print(_data.shape)  # (7,5) =>  oneHot 되어 column이 5개로 증가
print(type(_data))
print(_data.dtype)

x_data = _data[:6, ]    # (6,5)     hihell 부분
y_data = _data[1:, ]    # (6,5)     ihello 부분
y_data = np.argmax(y_data, axis=1)

print(x_data)
print(y_data)

x_data = x_data.reshape(1,6,5)  # (1,6,5)
y_data = y_data.reshape(1,6)

print(x_data.shape) # (1,6,5)
print(x_data.dtype)
print(y_data.shape) # (1,6)

# 데이터 구성
# x : (batch_size, sequeqnce_length, input_dim) 1,6,5
# 첫번째 아웃풋 : hidden_size = 2
# 첫번째 결과 : 1,6,5

num_classes = 5 
batch_size = 1          # (전체행)
sequence_length = 6     # 컬럼
input_dim = 5           # 몇개씩 작업
hidden_size = 5         # 첫번째 노드 출력 개수
learning_rate = 0.1


# X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])        # (?, 6, 5)
# Y = tf.placeholder(tf.int32, [None, sequence_length])                     # (? 6)

X = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, input_dim])        # (?, 6, 5)
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length])                     # (?, 6)
print(X)
print(Y)

# 2. 모델 구성
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)        #, state_is_tuple=True) #   cnffurtkdlwm
# initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, # initial_state=initial_state,
                                        dtype=tf.float32)
# outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
print(outputs)
print(outputs.shape)        # (1,6,5)

# FC layer
# X_for_fc = tf.reshape(outputs, [-1, hidden_size])   # (6,5)
# print(X_for_fc)
# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

###################################
# W, loss, train, prediction
##################################

weights = tf.ones([batch_size, sequence_length])        # 임의로 1을 넣는다.

sequence_loss = tf.contrib.seq2seq.sequence_loss(
                        logits=outputs, targets=Y, weights=weights
)
loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i, "loss:", l, "prediction:", result, "true Y:", y_data)
        # print(sess.run(weights))

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nprediction str :", ''.join(result_str))


# 위 모델의 문제점 :  train / test 분리가 안되어있다 => epoch 수가 많아지면 적합하게 된다(과적합 발생 가능)``