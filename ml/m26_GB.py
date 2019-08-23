from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
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
    model = GradientBoostingRegressor(max_depth=10)
    return model

def create_hyperparameters():
    loss = ['ls', 'lad', 'huber', 'quantile']
    criterion=["friedman_mse", "mse", "mae"]
    learning_rate = [0.1, 0.2, 0.3, 0.4]
    n_estimators = [100, 200, 300, 400, 500]
    max_depth = [200, 400, 600, 800, 1000]
    min_samples_leaf = [10,35,50,75,100]
    max_leaf_nodes = [1000, 2000, 3000, 4000, 5000]
    return {"loss":loss, "criterion":criterion, "learning_rate":learning_rate, "n_estimators":n_estimators, "max_depth":max_depth, 
            "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes}      # Map 형태로 반환

kfold = KFold(n_splits=5, shuffle=True)
model = build_model()
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=-1, cv=kfold)
search.fit(train_x, train_y)

predict_y = search.predict(test_x)

print(search.refit)
print("best_parameters : ",search.best_params_)
print("best_estimator : ",search.best_estimator_)
print("best_score(x_train) : ",search.best_score_)
# 평가하기
print("test_score(r2) :", search.score(test_x, test_y))

'''
True
best_parameters :  {'n_estimators': 100, 'min_samples_leaf': 75, 'max_leaf_nodes': 2000, 'max_depth': 600, 'loss': 'ls', 'learning_rate': 0.1, 'criterion': 'friedman_mse'}
best_estimator :  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=600,
             max_features=None, max_leaf_nodes=2000,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=75, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=100,
             n_iter_no_change=None, presort='auto', random_state=None,
             subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
             warm_start=False)
best_score(x_train) :  0.9348359771378902
test_score(r2) : 0.916578782915225
'''

