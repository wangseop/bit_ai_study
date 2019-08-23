# gridSearchCV 코드를 RandomizedSearchCV 코드로 변환
'''
k-fold에서는 train data set에 train 과 validation data가 서로 번걸아가면서 
'''
'''
pipeline의 장점 : 1. model을 fit할때 마다 전처리가 된다. 2. train / validation의 분리
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기
# warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size = 0.8, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

'''
방 법 1
'''
pipe = Pipeline([("scalar", MinMaxScaler()), ('svm', SVC())])

'''
방 법 2
'''
# from sklearn.pipeline import make_pipeline
# pipe = make_pipeline(MinMaxScaler(), SVC(C=100))


# └ SVC 모델 동작 전에 Data를 전처리한 후 모델에 fitting 시킨다.
#  ┗ pipeline으로 모델을 연결해줄때 파라미터의 이름은 모델의 이름을 명시적으로 붙여줘야한다 -> svc__
pipe.fit(x_train, y_train)

print("테스트 점수 : ", pipe.score(x_test, y_test))