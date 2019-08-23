import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings('ignore')
# ┗Error 메세지 출력을 무시하기 위해 ignore 문자열을 줌

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength","PetalWidth"]]

# classifier 알고리즘 모두 추출하기 --- (*1)
# warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')

# K-분할 크로스 밸리데이션 전용 객체
for i in range(3, 11):
    print('n_splits value :', i)
    kfold_cv = KFold(n_splits=i, shuffle=True)
    for (name, algorithm) in allAlgorithms:
        # 각 알고리즘 객체 생성하기
        clf = algorithm()

        # score 메서드를 가진 클래스를 대상으로 하기
        scores_mean = {}
        if hasattr(clf, "score"):

            # 크로스 밸리데이션
            scores = cross_val_score(clf, x, y, cv=kfold_cv)
            print(name,"의 정답률=")
    print("="*100)

'''
n_splits value : 3

LinearDiscriminantAnalysis 의 정답률=
0.98

MLPClassifier 의 정답률=
0.9733333333333333

QuadraticDiscriminantAnalysis 의 정답률=
0.9733333333333333

====================================================================================================
n_splits value : 4

LinearDiscriminantAnalysis 의 정답률=
0.9733285917496444

QuadraticDiscriminantAnalysis 의 정답률=
0.9731507823613087

LabelSpreading 의 정답률=
0.966927453769559
====================================================================================================
n_splits value : 5

LinearDiscriminantAnalysis 의 정답률=
0.9800000000000001

MLPClassifier 의 정답률=
0.9800000000000001

KNeighborsClassifier 의 정답률=
0.9733333333333333

====================================================================================================
n_splits value : 6

LinearDiscriminantAnalysis 의 정답률=
0.98

SVC 의 정답률=
0.98

MLPClassifier 의 정답률=
0.9733333333333333


====================================================================================================
n_splits value : 7

SVC 의 정답률=
0.9802102659245516

QuadraticDiscriminantAnalysis 의 정답률=
0.9799010513296228

MLPClassifier 의 정답률=
0.9730983302411874

====================================================================================================
n_splits value : 8

LinearDiscriminantAnalysis 의 정답률=
0.9802631578947367

MLPClassifier 의 정답률=
0.9736842105263157

QuadraticDiscriminantAnalysis 의 정답률=
0.9736842105263157

====================================================================================================
n_splits value : 9


MLPClassifier 의 정답률=
0.9803921568627452

LinearDiscriminantAnalysis 의 정답률=
0.979983660130719

QuadraticDiscriminantAnalysis 의 정답률=
0.979983660130719

====================================================================================================
n_splits value : 10

QuadraticDiscriminantAnalysis 의 정답률=
0.9800000000000001

LinearDiscriminantAnalysis 의 정답률=
0.9800000000000001

MLPClassifier 의 정답률=
0.9733333333333334




'''