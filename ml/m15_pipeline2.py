# m11_randomSearch.py 에 pipeline을 적용하시오

# gridSearchCV 코드를 RandomizedSearchCV 코드로 변환

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

# 그리드 서치에서 사용할 매개 변수 --- (*1)
parameters = {      # SVM
    "svc__C": [1,10,100,1000], "svc__kernel":["linear", "rbf", "sigmoid"], "svc__gamma":[0.001, 0.0001]
}

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 그리드 서치 --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
# pipe = Pipeline([("scalar", MinMaxScaler()), ('svm', SVC())])
#  ┗ pipeline으로 모델을 연결해줄때 파라미터의 이름은 모델의 이름을 명시적으로 붙여줘야한다 -> svm__
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(), SVC(C=100))
#  ┗ pipeline으로 모델을 연결해줄때 파라미터의 이름은 모델의 이름을 명시적으로 붙여줘야한다 -> svc__

clf = RandomizedSearchCV(estimator=pipe, param_distributions=parameters, cv=kfold_cv,  n_iter=10, n_jobs=1, verbose=1)      # SVM

#clf.fit(x, y)
clf.fit(x_train, y_train)

print("최적의 매개 변수 =", clf.best_estimator_)

# 최적의 매개 변수로 평가하기 --- (*3)
y_pred = clf.predict(x_test)
print("최종 정답률 =", accuracy_score(y_test, y_pred))
# last_score = clf.score(x_test, y_test)
# print("최종 정답률 =", last_score)


'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:    0.1s finished
최적의 매개 변수 = Pipeline(memory=None,
     steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svc', SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
'''