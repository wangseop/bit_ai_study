from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding='utf-8')

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df['연'] <= 2015)
test_year = (df['연'] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []  # 학습 데이터
    y = []  # 결과
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

# 직선 회귀 분석하기

def build_model():
    model = RandomForestRegressor(max_depth=10)
    return model

def create_hyperparameters():
    criterion = ["mse", "mae"]
    n_estimators = [225, 1000, 1500]
    max_depth = [1600]
    min_samples_leaf = [3]
    max_leaf_nodes = [100, 125]
    n_jobs = [-1]
    return {"criterion":criterion,"n_estimators":n_estimators, "max_depth":max_depth, 
            "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes,
            "n_jobs":n_jobs}      # Map 형태로 반환

kfold = KFold(n_splits=5, shuffle=True)
model = build_model()
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=-1, cv=kfold)
search.fit(train_x, train_y)

predict_y = search.predict(test_x)

print("best_parameters : ",search.best_params_)
print("best_estimator : ",search.best_estimator_)
print("best_score(x_train) : ",search.best_score_)
# 평가하기
print("test_score(r2) :", search.score(test_x, test_y))

'''
best_parameters :  {'n_jobs': -1, 'n_estimators': 250, 'min_samples_leaf': 3, 'max_leaf_nodes': 100, 'max_depth': 200, 'criterion': 'mse'}
best_estimator :  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=200,
           max_features='auto', max_leaf_nodes=100,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=3, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
best_score(x_train) :  0.9380923115318133
test_score(r2) : 0.9175293818620752
'''

