from keras.datasets import mnist, cifar10, boston_housing
from sklearn.datasets import load_boston, load_breast_cancer

import numpy as np
import pandas as pd

# datasets load
pima_data = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=',')
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8', header=None)
wine = pd.read_csv("./data/winequality-white.csv", sep=';', encoding='utf-8')

(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
(cifar10_train_x, cifar10_train_y),(cifar10_test_x, cifar10_test_y) = cifar10.load_data()
(boston_train_x, boston_train_y), (boston_test_x, boston_test_y) = boston_housing.load_data()

cancer = load_breast_cancer()



####### data save #######

#### pima data ####
## pima.npy (전체 dataset, (768, 9))
## pima_x.npy (전체 x_dataset, (768, 8))
## pima_y.npy (전체 y_dataset, (768, ))
'''
np.save('./data/pima.npy', pima_data)  
np.save('./data/pima_x.npy', pima_data[:,:-1])
np.save('./data/pima_y.npy', pima_data[:,-1:])

pima_load = np.load('./data/pima.npy')
pima_x_load = np.load('./data/pima_x.npy')
pima_y_load = np.load('./data/pima_y.npy')

print(pima_load)
print(pima_x_load)
print(pima_y_load)

print(pima_load.shape)
print(pima_x_load.shape)
print(pima_y_load.shape)
'''

#### iris data ####
## iris.npy (전체 dataset, (150, 5))
## iris_x.npy (전체 x_dataset, (150, 4))
## iris_y.npy (전체 y_dataset, (150, ))
'''
iris_data_np = iris_data.values
numlist = []
for i in range(len(iris_data.values)):
    if iris_data.values[i][-1] == 'Iris-setosa':
        numlist.append(0)
    elif iris_data.values[i][-1] == 'Iris-versicolor':
        numlist.append(1)
    else:
        numlist.append(2)

numlist = np.array(numlist)
numlist = numlist.reshape((150, 1))
iris_data_np[:, [-1]] = numlist
np.save('./data/iris.npy', iris_data_np)
np.save('./data/iris_x.npy', iris_data_np[:,:-1])
np.save('./data/iris_y.npy', iris_data_np[:,-1])

iris_load = np.load('./data/iris.npy')
iris_x_load = np.load('./data/iris_x.npy')
iris_y_load = np.load('./data/iris_y.npy')

print(iris_load)
print(iris_x_load)
print(iris_y_load)

print(iris_load.shape)
print(iris_x_load.shape)
print(iris_y_load.shape)

#### wine data ####
## wine.npy (전체 dataset, (4898, 12))
## wine_x.npy (전체 x_dataset, (4898, 11))
## wine_y.npy (전체 y_dataset, (4898, ))

np.save('./data/wine.npy', wine.values)
np.save('./data/wine_x.npy', wine.values[:,:-1])
np.save('./data/wine_y.npy', wine.values[:,-1])

wine_load = np.load('./data/wine.npy')
wine_x_load = np.load('./data/wine_x.npy')
wine_y_load = np.load('./data/wine_y.npy')

print(wine_load)
print(wine_x_load)
print(wine_y_load)

print(wine_load.shape)
print(wine_x_load.shape)
print(wine_y_load.shape)
'''
'''
#### mnist data ####
## mnist_train_x.npy (전체 x_train_dataset, (60000, 28, 28))
## mnist_test_x.npy (전체 x_test_dataset, (10000, 28, 28))
## mnist_train_y.npy (전체 y_train_dataset, (60000, ))
## mnist_test_y.npy (전체 y_test_dataset, (10000, ))
'''
'''
np.save('./data/mnist_train_x.npy', mnist_train_x)
np.save('./data/mnist_test_x.npy', mnist_test_x)
np.save('./data/mnist_train_y.npy', mnist_train_y)
np.save('./data/mnist_test_y.npy', mnist_test_y)

mnist_train_x_load = np.load('./data/mnist_train_x.npy')
mnist_test_x_load = np.load('./data/mnist_test_x.npy')
mnist_train_y_load = np.load('./data/mnist_train_y.npy')
mnist_test_y_load = np.load('./data/mnist_test_y.npy')

print(mnist_train_x_load)
print(mnist_test_x_load)
print(mnist_train_y_load)
print(mnist_test_y_load)

print(mnist_train_x_load.shape)
print(mnist_test_x_load.shape)
print(mnist_train_y_load.shape)
print(mnist_test_y_load.shape)
'''


#### cifar10 data ####
## cifar10_train_x.npy (전체 x_train_dataset, (50000, 32, 32, 3))
## cifar10_test_x.npy (전체 x_test_dataset, (10000, 32, 32, 3))
## cifar10_train_y.npy (전체 y_train_dataset, (50000, 1))
## cifar10_test_y.npy (전체 y_test_dataset, (10000, 1))
'''
np.save('./data/cifar10_train_x.npy', cifar10_train_x)
np.save('./data/cifar10_test_x.npy', cifar10_test_x)
np.save('./data/cifar10_train_y.npy', cifar10_train_y)
np.save('./data/cifar10_test_y.npy', cifar10_test_y)

cifar10_train_x_load = np.load('./data/cifar10_train_x.npy')
cifar10_test_x_load = np.load('./data/cifar10_test_x.npy')
cifar10_train_y_load = np.load('./data/cifar10_train_y.npy')
cifar10_test_y_load = np.load('./data/cifar10_test_y.npy')

print(cifar10_train_x_load)
print(cifar10_test_x_load)
print(cifar10_train_y_load)
print(cifar10_test_y_load)

print(cifar10_train_x_load.shape)
print(cifar10_test_x_load.shape)
print(cifar10_train_y_load.shape)
print(cifar10_test_y_load.shape)
'''
#### boston data ####
## boston_train.npy (전체 train_dataset, (404,14))
## boston_test.npy (전체 test_dataset, (102,14))
## boston_train_x.npy (전체 x_train_dataset, (404,13))
## boston_test_x.npy (전체 x_testdataset, (102,13))
## boston_train_y.npy (전체 y_train_dataset, (404,))
## boston_test_y.npy (전체 y_test_dataset, (102,))

np.save('./data/boston_train_x.npy', boston_train_x)
np.save('./data/boston_test_x.npy', boston_test_x)
np.save('./data/boston_train_y.npy', boston_train_y)
np.save('./data/boston_test_y.npy', boston_test_y)

np.save('./data/boston_train.npy', np.concatenate((boston_train_x, boston_train_y.reshape(boston_train_y.shape[0], 1)), axis=1))
np.save('./data/boston_test.npy', np.concatenate((boston_test_x, boston_test_y.reshape(boston_test_y.shape[0], 1)), axis=1))



boston_train_load = np.load('./data/boston_train.npy')
boston_test_load = np.load('./data/boston_test.npy')
boston_train_x_load = np.load('./data/boston_train_x.npy')
boston_test_x_load = np.load('./data/boston_test_x.npy')
boston_train_y_load = np.load('./data/boston_train_y.npy')
boston_test_y_load = np.load('./data/boston_test_y.npy')

print(boston_train_load)
print(boston_test_load)
print(boston_train_x_load)
print(boston_test_x_load)
print(boston_train_y_load)
print(boston_test_y_load)

print(boston_train_load.shape)
print(boston_test_load.shape)
print(boston_train_x_load.shape)
print(boston_test_x_load.shape)
print(boston_train_y_load.shape)
print(boston_test_y_load.shape)


#### breast_cancer data ####
## breast_cancer.npy (전체 dataset, (569,31))
## breast_cancer_x.npy (전체 x_dataset, (569,30))
## breast_cancer_y.npy (전체 y_dataset, (569,))

np.save('./data/breast_cancer.npy', np.concatenate((cancer.data, cancer.target.reshape(cancer.target.shape[0], 1)), axis=1))
np.save('./data/breast_cancer_x.npy', cancer.data)
np.save('./data/breast_cancer_y.npy', cancer.target)


breast_cancer_load = np.load('./data/breast_cancer.npy')
breast_cancer_x_load = np.load('./data/breast_cancer_x.npy')
breast_cancer_y_load = np.load('./data/breast_cancer_y.npy')

print(breast_cancer_load)
print(breast_cancer_x_load)
print(breast_cancer_y_load)

print(breast_cancer_load.shape)
print(breast_cancer_x_load.shape)
print(breast_cancer_y_load.shape)




