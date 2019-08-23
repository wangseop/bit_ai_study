import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier         # RandomForest -> XG_boost(pip install XG_boost) -> keras
from sklearn.metrics import accuracy_score                  # 분류 score 매기는 것
from sklearn.metrics import classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop("quality", axis=1)

# y 레이블 변경하기 --- (*2)
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
                                        x, y, test_size=0.2                                    
)


def build_model():
    model = RandomForestClassifier(max_depth=10)
    return model

def create_hyperparameters():
    n_estimators = [100, 200, 400, 800, 1000]
    max_depth = [100, 200,300,400,500]
    min_samples_leaf = [1, 5, 10]
    max_leaf_nodes = [100, 1000, 2000]
    n_jobs = [-1]
    return {"n_estimators":n_estimators, "max_depth":max_depth, 
            "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes,
            "n_jobs":n_jobs}      # Map 형태로 반환

kfold = KFold(n_splits=5, shuffle=True)
# 학습하기
model = build_model()
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=-1, cv=kfold)
search.fit(x_train, y_train)

print("best_parameters : ",search.best_params_)
print("best_estimator : ",search.best_estimator_)
print("best_score(x_train) : ",search.best_score_)


# 평가하기
print("test_score :", search.score(x_test, y_test))

'''
best_parameters :  {'n_jobs': -1, 'n_estimators': 100, 'min_samples_leaf': 1, 'max_leaf_nodes': 1000, 'max_depth': 100}
best_estimator :  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=1000,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
best_score(x_train) :  0.9369576314446146
test_score : 0.9357142857142857
'''