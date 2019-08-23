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
    model = DecisionTreeClassifier(max_depth=10)
    return model

def create_hyperparameters():
    max_depth = [10,20,30,40,50]
    min_samples_leaf = [1, 10, 100]
    max_leaf_nodes = [10, 100, 1000]
    return {"max_depth":max_depth, "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes}      # Map 형태로 반환

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
#└ 여기서는 model의 score와 accuracy_score와 점수가 동일하게 나오는데, 이는 현재 모델이 분류모델이기에 데이터 값이 분류형태로 나왔기 때문이다.


# 실습 acc 66%를 70% 이상으로 올리기.