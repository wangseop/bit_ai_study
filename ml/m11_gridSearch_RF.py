import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기
# warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size = 0.8, shuffle=True)

# 그리드 서치에서 사용할 매개 변수 --- (*1)
parameters = [      # RandomForestClassifier
    {"n_estimators": [10,100,300,500], "max_features":["auto"], "class_weight":["balanced", "balanced_subsample", None]},
    {"n_estimators": [10,100,300,500], "max_features":["sqrt"], "class_weight":["balanced", "balanced_subsample", None]},
    {"n_estimators": [10,100,300,500], "max_features":["log2"], "class_weight":["balanced", "balanced_subsample", None]}
]

# 그리드 서치 --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv)      # RandomForestClassifier
# ┗ 모델과 모델에 맞는 파라미터들을 수정해주면 된다
clf.fit(x_train, y_train)
print("최적의 매개 변수 =", clf.best_estimator_)

# 최적의 매개 변수로 평가하기 --- (*3)
y_pred = clf.predict(x_test)
print("최종 정답률 =", accuracy_score(y_test, y_pred))
# last_score = clf.score(x_test, y_test)
# print("최종 정답률 =", last_score)

'''
최적의 매개 변수 = RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='sqrt',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=None, oob_score=False,
            random_state=None, verbose=0, warm_start=False)
최종 정답률 = 0.9666666666666667

'''