from xgboost import XGBRegressor
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
    model = XGBRegressor()
    return model

def create_hyperparameters():
    learning_rate = [0.1, 0.2, 0.3, 0.4]
    n_estimators = [200, 400, 600, 800, 1000]
    max_depth = [500, 1000, 1500, 2000]
    min_samples_leaf = [50,75,100, 125, 150]
    max_leaf_nodes = [2000, 4000, 6000, 8000, 10000]
    n_jobs=[-1]
    return {"learning_rate":learning_rate, "n_estimators":n_estimators, "max_depth":max_depth, 
            "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes, "n_jobs":n_jobs}      # Map 형태로 반환

kfold = KFold(n_splits=5, shuffle=True)

model = build_model()
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=20, n_jobs=-1, cv=kfold)

search.fit(train_x, train_y)

predict_y = search.predict(test_x)

print(search.refit)
print("best_parameters : ",search.best_params_)
print("best_estimator : ",search.best_estimator_)
print("best_score(x_train) : ",search.best_score_)
# 평가하기
print("test_score(r2) :", search.score(test_x, test_y))



'''
best_parameters :  {'n_jobs': -1, 'n_estimators': 1000, 'min_samples_leaf': 50, 'max_leaf_nodes': 6000, 'max_depth': 1000, 'learning_rate': 0.1}
best_estimator :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=1000, max_leaf_nodes=6000, min_child_weight=1,
       min_samples_leaf=50, missing=None, n_estimators=1000, n_jobs=-1,
       nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
       subsample=1, verbosity=1)
best_score(x_train) :  0.9265507874701076
test_score(r2) : 0.9092304101013358
'''