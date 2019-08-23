from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()
print(boston.data.shape)
print(boston.keys())
print(boston.target)
print(boston.target.shape)

x = boston.data
y = boston.target

print(type(boston))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LinearRegression, Ridge, Lasso # Ridge, Lasso
# 모델 완성하시오.

model = LinearRegression()
model2 = Ridge()
model3 = Lasso()

model.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)

score = model.score(x_test, y_test)
score2 = model2.score(x_test, y_test)
score3 = model3.score(x_test, y_test)


y_pred = model.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)

print("score(LinearRegressor) : %.4f" % score)
print("score(Ridge) : %.4f" % score2)
print("score(Lasso) : %.4f" % score3)
# print('y_pred :', y_pred)