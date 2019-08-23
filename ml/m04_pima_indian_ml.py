from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
#model = LinearSVC()
#model = SVC()
model = KNeighborsClassifier(n_neighbors=1)

# 모델 실행
model.fit(x, y)

y_predict = model.predict(x)

# 결과 출력
print('\n Accuracy: %.4f'%(accuracy_score(y, y_predict)))