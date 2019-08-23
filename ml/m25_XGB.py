import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBClassifier
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
    model = XGBClassifier(max_depth=10, n_jobs=4)
    return model

def create_hyperparameters():
    learning_rate = [0.1, 0.2, 0.3, 0.4]
    n_estimators = [100, 200, 300, 400, 500]
    max_depth = [200, 400, 600, 800, 1000]
    min_samples_leaf = [10,35,50,75,100]
    max_leaf_nodes = [1000, 2000, 3000, 4000, 5000]
    n_jobs=[-1]
    min_child_weight=[1,2,3,4,5]
    return {"learning_rate":learning_rate, "n_estimators":n_estimators, "max_depth":max_depth, 
            "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes, "n_jobs":n_jobs,
            "min_child_weight":min_child_weight}      # Map 형태로 반환

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
best_parameters :  {'n_jobs': -1, 'n_estimators': 200, 'min_samples_leaf': 20, 'max_leaf_nodes': 1000, 'max_depth': 400, 'learning_rate': 0.3}
best_estimator :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.3,
       max_delta_step=0, max_depth=400, max_leaf_nodes=1000,
       min_child_weight=1, min_samples_leaf=20, missing=None,
       n_estimators=200, n_jobs=-1, nthread=None,
       objective='multi:softprob', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)
best_score(x_train) :  0.9389994895354773
test_score : 0.95
'''