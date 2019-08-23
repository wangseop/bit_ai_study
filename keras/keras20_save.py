#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
model = Sequential()        # 순차적인 모델

# 모델링 과정
# model.add(Dense(10, input_dim = 2, activation='relu'))   # input 1 / output 5 (input layer)
model.add(Dense(30, input_shape = (3, ), activation='relu', 
                kernel_regularizer=regularizers.l1(0.0005)))   # input 1 / output 5 (input layer)   # input_shape = (1, ) -> (2, )
# model.add(Dropout(0.01))
# model.add(BatchNormalization())
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(22))
model.add(Dense(20))
model.add(Dense(17))
model.add(Dense(12))
model.add(Dense(1))     # 1 -> 2

# model.summary()
model.save('savetest01.h5')
print('save complete.')