from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


# 1. 데이터
x_data = [[0,0], [1, 0], [0,1], [1, 1]]
y_data = [0,1,1,0]

# 2. 모델
# model = LinearSVC()
# └ LinearSVC는 선 1개로 분류할 수 없는 데이터의 경우에는 accuracy가 1이 나오지 않는다
model = SVC()
# └ 따라서 차원을 늘려 선이 아닌 면 혹은 그 이상으로 구분지어줄 매개체를 선택해야한다

# 3. 실행
model.fit(x_data, y_data)

# 4. 평가 예측
x_test = [[0,0], [1, 0], [0,1], [1, 1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 :", y_predict)
print("acc =", accuracy_score([0,1,1,0], y_predict))
# └ xor는 선 1개로 분류할 수 없다. => 


