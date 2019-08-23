#1. 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 모델링 과정
model.add(Dense(5, input_dim = 1, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
print("                << Current Model >>")
model.summary()
#3. 훈련
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
# optimizers = [
#                 ['Adam', Adam(lr=0.01)], 
#                 ['SGD', SGD(lr=0.01)], 
#                 ['RMSprop', RMSprop(lr=0.01)], 
#                 ['Adagrad', Adagrad(lr=0.01)], 
#                 ['Adadelta', Adadelta(lr=0.01)], 
#                 ['Adamax', Adamax(lr=0.01)], 
#                 ['Nadam', Nadam(lr=0.01)]
#             ]
optimizers = [
                ['Adam', Adam()],              # Adam       default Learning Rate : 0.001
                ['SGD', SGD()],                # SGD        default Learning Rate : 0.01
                ['RMSprop', RMSprop()],        # RMSprop    default Learning Rate : 0.001 
                ['Adagrad', Adagrad()],        # Adagrad    default Learning Rate : 0.01 
                ['Adadelta', Adadelta()],      # Adadelta   default Learning Rate : 1.0 
                ['Adamax', Adamax()],          # Adamax     default Learning Rate : 0.002 
                ['Nadam', Nadam()]             # Nadam      default Learning Rate : 0.002
            ]
print("                << Optimizer Test >>")

for i in range(len(optimizers)):
    # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    loss = "mse"
    metrics = "mse"
    model.compile(loss=loss, optimizer=optimizers[i][1], metrics=[metrics])
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)       # 훈련(실행)시키다

    #4. 평가 예측
    mse, _ = model.evaluate(x, y, batch_size=1)
    print("#### ", optimizers[i][0], "(loss :", loss, "metrics : ", metrics, ") ####")
    print()
    print("epochs =", 100, " / ", "batch_size =", 1)
    print('mse : ', mse)
    print()
    x_test = np.array([1.5,2.5,3.5])
    pred1 = model.predict(x_test)
    print("Test value :", x_test)
    print("predict :", pred1)

"""

                << Current Model >>
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0

=================================================================

                << Optimizer Test >> (Learning Rate = 0.01)

    ####  Adam (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  1.1546319456101628e-14

Test value : [1.5 2.5 3.5]
predict : [[1.4999999]
 [2.5000002]
 [3.4999998]]

--------------------------------------------------------------
--------------------------------------------------------------

    ####  SGD (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  3.552713678800501e-15

Test value : [1.5 2.5 3.5]
predict : [[1.5      ]
 [2.5000002]
 [3.4999995]]

--------------------------------------------------------------
--------------------------------------------------------------

    ####  RMSprop (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.127509786747396

Test value : [1.5 2.5 3.5]
predict : [[1.2555085]
 [2.1594486]
 [3.0633886]]

--------------------------------------------------------------
--------------------------------------------------------------

    ####  Adagrad (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.0

Test value : [1.5 2.5 3.5]
predict : [[1.4999999]
 [2.5      ]
 [3.5      ]]

--------------------------------------------------------------
--------------------------------------------------------------

    ####  Adadelta (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.0

Test value : [1.5 2.5 3.5]
predict : [[1.4999999]
 [2.5      ]
 [3.5      ]]

--------------------------------------------------------------
--------------------------------------------------------------

    ####  Adamax (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.0

Test value : [1.5 2.5 3.5]
predict : [[1.4999999]
 [2.5      ]
 [3.5      ]]

--------------------------------------------------------------
--------------------------------------------------------------

    ####  Nadam (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.0

Test value : [1.5 2.5 3.5]
predict : [[1.4999999]
 [2.5      ]
 [3.5      ]]

"""

"""
                << Current Model >>
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
                << Optimizer Test >>  (Learning Rate = default)

####  Adam (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.0747408790193731

Test value : [1.5 2.5 3.5]
predict : [[1.8191746]
 [2.5875168]
 [3.355859 ]]

####  SGD (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  3.4022562545032997e-10

Test value : [1.5 2.5 3.5]
predict : [[1.5000224]
 [2.5000074]
 [3.4999926]]

####  RMSprop (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  4.0392555522572593e-07

Test value : [1.5 2.5 3.5]
predict : [[1.4997978]
 [2.4994762]
 [3.4991543]]

####  Adagrad (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  5.773159728050814e-14

Test value : [1.5 2.5 3.5]
predict : [[1.5000002]
 [2.5000002]
 [3.5000005]]

####  Adadelta (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  4.709582128725742e-07

Test value : [1.5 2.5 3.5]
predict : [[1.4997417]
 [2.5003192]
 [3.500896 ]]

####  Adamax (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  4.369539396975597e-09

Test value : [1.5 2.5 3.5]
predict : [[1.4998629]
 [2.4999287]
 [3.499994 ]]

####  Nadam (loss : mse metrics :  mse ) ####

epochs = 100  /  batch_size = 1
mse :  0.00013213982310844585

Test value : [1.5 2.5 3.5]
predict : [[1.4896158]
 [2.488189 ]
 [3.4875243]]
"""