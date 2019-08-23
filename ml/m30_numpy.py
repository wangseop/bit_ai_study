import numpy as np
a = np.arange(10)
print(a)
np.save('aaa.npy', a)   # 저장
b = np.load('aaa.npy')  # 불러오기
print(b)


#### 모델 저장하기 ####
# model.save('savetest01.h5')

#### 모델 불러오기 ####
# from keras.models import load_model
# model = load_model("savetest01.h5")
# from keras.layers import Dense
# model.add(Dense(1))



#### pandas를 numpy로 바꾸기 ####
# =>판다스.value

#### csv 불러오기 ####
'''
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=',')
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8')
            # index_col = 0, encoding='cp949', sep=',', header=None
            # names=['x1','x2','x3','x4','y']
             # └header=None은 컬럼명 없이 가져오는것
wine = pd.read_csv("./data/winequality-white.csv", sep=',', encoding='utf-8')
'''


#### utf-8 ####
'''
#-*- coding: utf-8 -*-
# => python 기본 인코딩은 ascii 인데 인터프리터가 .py 파일을 해석할 때 ascii 방식이 아닌 언어도 ascii 방식으로 해석하려 하므로
#   Syntax Error 가 발생할 수 있다.
#   이를 해결하기 위해서 스크립트 파일 첫줄에 -*- coding: utf-8 -*- 를 추가해주면 파일의 인코딩 방식을 안에 쓰인 coding 방식으로 지정하여 해석
'''



#### 한글 처리 ####
'''
1. 파일 맨 위 주석을 삽입한다. (#-*- coding:utf-8 -*- 혹은 #-*- coding:cp949 -*-)

2. utf-8 혹은 UNICODE로 디코딩 후 다시 인코딩 
 : 파이썬의 문자열 인식 default 방식은 Unicode이다.
   문자열마다 encoding된 방식이 다르므로 파이썬에서 인식하도록 하려면 decoding('현재encoding') 해주고 encoding('원하는 방식') 해줘야한다.
   만일 파이썬에서 a 라는 문자열이 인식되지 않는다면 [ a.decode('cp949').encode('utf-8') ] 로 해주면 한글이 제대로 인식될 것이다.

3. 외부 라이브러리의 기본 인코딩 방식 설정
 : 끌어온 라이브러리의 한글이 많다면, 해당 라이브러리 전체를 재인코딩해주면 된다.
 import sys

 reload(sys)
 
 sys.setdefaultencoding('utf-8')
 의 방식으로 해주면 끌어온 라이브러리 전체의 인코딩 방식을 변경할 수 있다.
'''

#### 각종 샘플 데이터 셋 ####
'''
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

from keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())        # data, target
# boston.data   : x값, 넘파이
# boston.target : y값, 넘파이

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
'''

# csv 저장 #
'''
1. pandas DataFrame의 to_csv 메서드 이용
    : 저장할 array 형태의 data를 dataFrame에 넣어주고( [ pandas.DataFrame(array_data) ] ) 
     해당 객체의 [ to_csv(path_or_buf='저장할 csv 파일 경로', sep=',', header='True',index='True', encoding='utf-8'...) ]

     사용 예)
     ====================================================================
     import pandas as pd
     
     data = [[1,2,3,4],[5,6,7,8]]

     dataframe = pd.DataFrame(data)
     dataframe.to_csv("저장할 csv파일 경로", header=False, index=False)
     ====================================================================

2. csv 라이브러리를 활용한다
    
    사용 예)
    =======================================================
    import csv

    data = [[1,2,3,4],[5,6,7,8]]

    csvfile = open("저장할 csv 파일 경로", "w", newline="")
    csvwriter = csv.writer(csvfile)
    for row in data:
        csvwriter.writerow(row)
    csvfile.close()
    =======================================================
'''


# 1. 예제 데이터를 npy로 저장
# 2. x와 y의 shape를 표시
# 3. m30_numpy_sample.py
# 4. 한글처리 찾을것
# 5. csv 저장