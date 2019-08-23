import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier         # RandomForest -> XG_boost(pip install XG_boost) -> keras
from sklearn.metrics import accuracy_score                  # 분류 score 매기는 것
from sklearn.metrics import classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop("quality", axis=1)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
                                        x, y, test_size=0.2                                    
)

# 학습하기
model = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=123456, max_features=3)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# 평가하기
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))
print("Score :", score)
#└ 여기서는 model의 score와 accuracy_score와 점수가 동일하게 나오는데, 이는 현재 모델이 분류모델이기에 데이터 값이 분류형태로 나왔기 때문이다.


# 실습 acc 66%를 70% 이상으로 올리기.