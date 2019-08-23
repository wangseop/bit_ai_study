from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

#  seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
# dataset = numpy.loadtxt("D:\\study\\data\\pima-indians-diabetes.csv", delimiter=",")
x = dataset[:, 0:8]
# ‪X = dataset[:, 0:8]
y = dataset[:, 8]

# 모델의 설정
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# output에 sigmoid를 주게 되면 결과치를 0이나 1로 근사하게 도출한다

#  모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# sigmoid 결과로부터 0이나 1로 분류하기 위해서 loss를 binary_crossentropy 사용

# 모델 실행
model.fit(x, y, epochs=300, batch_size=10)

# 결과 출력
print('\n Accuracy: %.4f'%(model.evaluate(x, y)[1]))