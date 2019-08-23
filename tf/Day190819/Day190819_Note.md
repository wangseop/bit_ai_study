## Tensorflow Session
* Session은 Tensor Model 안에 들어가는  Graph를 해석 할 수 있다.
* 즉 Compile에 대한 Graph, fit에 대한 Graph, evaluate/ predict에 대한 graph를 Session에서 run 하게 되면
* 그 그래프에 해당하는 내용이 출력된다. 
* Keras에서의 Compile, fit, evaluate, predic는 모두  run()을 통해 해석된다.
* Session의 run()으로 각 부분별 그래프 넣어 일반화 시킨것이 Keras Style이다.

## Tensorflow Session의 Variable
* Variable은 변수로 사용되는데, 변수를 이용하려면 global_variables_initializer 가 필요하다.
    * Variable이 가진 값을 사용하려면 Session의 run()  메서드를 이용해야 한다.
    * 이 방식을 사용하는 법은 총 3가지 존재한다.
        * 1) Tensorflow Session().run()
        * 2) Tensorflow InteractiveSession().eval()
        * 3) Tensorflow Session().run() => 현 그래프 지점이 w일때 w.eval(Session='해당세션')

## Placeholder
* placeholder는 data의 구조만 잡아두고 실제 data는 나중에 넣는 type으로 실제 data는 Session을 run하는 시점에서 입력해준다.
* 입력하기 위해서는 속성명을 명시해주고자 할 때엔 feed_dict={} 형태로 준다.

## Keras와 Model 구성의 차이점
* Keras와 다르게 learning할 모델 함수를 직접 구성한다 (=>hypothesis = X * W + b)
* compile에 필요한 속성값을 직접 구성한다. =>       loss 함수 :     cost = tf.reduce_sum(tf.square(hypothesis - y))
*                                                 optimizer 함수 : train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
* model의 fit을 위해서 Session을 열어 해당 Session을 run하여 fit 시켜준다.

## Multi Variable
* input_dim 이 커질 경우, weight가 각 input 열마다 적용되어야 한다.
* 각 열마다 weight를 곱해줘도 되지만, 개수가 많아질 수록 비효율적이다.
* 이를 효율적으로 하기 위해서 행렬식의 형태로 곱해서 일반화한다.
    * => hypothesis = tf.matmul(X, W) + b  의 형태로 matmul을 통해 행렬식의 곱형태로 계산
    * X * W 의 shape == b의 shape == hypothesis의 shape => hypothesis의 output 형태에 맞춰서 모양도 맞춰져야 한다.
    * X * W => X의 shape=(m,n), W의 shape=(o,p) 이면 n == o 가 되게끔 형태가 맞아야 한다.


##  Tensorflow 이진 분류 모델
* Hypothesis를 sigmoid 함수로 매핑(이진 분류도 크게 다중 분류의 일종이므로 softmax 매핑도 가능하다)
    * => 1. tf.sigmoid(tf.matmul(X, W) + b)
* cost (loss) 함수로 binary_crossentropy 사용하며, 이를 식으로 표현하면 
    * => 2. cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
* optimizer 함수와 cost 결합하여 train 그래프 생성 
    * => 3. train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
* 예측을 위해서 계산한 값이 0, 1 로 분류될 수 있도록 반올림 작업 수행
    * => 4. predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
* 예측한 값과 실제 Y 값을 같은 값인지를 기준으로 정확도 측정    
    * => 5. accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
* 각 과정에서 생성한 변수들은 모두 그래프 형태이므로 원하는 X,Y 에 따라 run 시켜주게 되면 해당하는 값을 얻을 수 있다.

##  Tensorflow 다중 분류 모델
* Hypothesis를 softmax 함수로 매핑
    * => 1. tf.softmax(tf.matmul(X, W) + b)
* cost (loss) 함수로 binary_crossentropy 사용하며, 이를 식으로 표현하면 
    * => 2. tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
* optimizer 함수와 cost 결합하여 train 그래프 생성 
    * => 3. train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
* 예측을 위해서 계산한 값이 0, 1 로 분류될 수 있도록 반올림 작업 수행
    * => 4. predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
* 예측한 값과 실제 Y 값을 같은 값인지를 기준으로 정확도 측정    
    * => 5. accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
* 각 과정에서 생성한 변수들은 모두 그래프 형태이므로 원하는 X,Y 에 따라 run 시켜주게 되면 해당하는 값을 얻을 수 있다.
* Tensorflow argmax는 다중 분류 모델에서 쓰기 유용한 함수이다.
    * tf.argmax(a, 1) => a array에서 각 행에서 가장 큰 값을 가지는 인덱스번호
    * tf.argmax(a, 1) 로 뽑히는 인덱스 값은 onehot encoding의 결과를 나타낸다
    * tf.argmax(a, 0) => a array 에서 각 열에서 가장 큰 값을 가지는 인덱스 번호 