import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


### npy load
train_x_num = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_x_num.npy')
train_x_cat = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_x_cat.npy')

test_x_num = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_x_num.npy')
test_x_cat = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_x_cat.npy')

train_id = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_id.npy')
test_id = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_id.npy')

train_y = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_y.npy')

### npy shape
print("train_x_num shape :", train_x_num.shape)
print("train_x_cat shape :", train_x_cat.shape)
print("test_x_num shape :", test_x_num.shape)
print("test_x_cat shape :", test_x_cat.shape)
print("train_id shape :", train_id.shape)
print("test_id shape :", test_id.shape)
print("train_y shape :", train_y.shape)

### string type OneHotEncoding
from sklearn.preprocessing import LabelEncoder

for col in range(train_x_cat.shape[1]):
    le = LabelEncoder()
    tmp_x = train_x_cat
    le.fit(list(train_x_cat[:, [col]]) + list(test_x_cat[:, [col]]))
    train_x_cat[:, [col]] = le.transform(train_x_cat[:, [col]]).reshape(-1, 1)
    test_x_cat[:, [col]] = le.transform(test_x_cat[:, [col]]).reshape(-1, 1)

train_x = np.concatenate((train_x_num, train_x_cat), axis=1)
test_x = np.concatenate((test_x_num, test_x_cat), axis=1)

print(train_x.shape)
print(test_x.shape)

### Scaler 적용

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()

scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/train_x.npy', train_x)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/test_x.npy', test_x)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/train_y.npy', train_y)


'''
df_train = pd.read_csv("./data/kaggle/categorical_feature_encoding_challenge/train.csv")
df_test = pd.read_csv("./data/kaggle/categorical_feature_encoding_challenge/test.csv")

# print(df_train.shape)   # (300000, 25)
# print(df_test_x.shape)  # (200000, 24)

train_size = 300000
test_size = 2000000

df_train_y = df_train['target']
df_train_x = df_train.drop(['target', 'id'], axis=1)

df_test_y = df_test['id']
df_test_x = df_test.drop(['id'], axis=1)


print(df_train_x.shape) # (300000, 23)
print(df_train_y.shape) # (300000, )
print(df_test_x.shape) # (200000, 23)
print(df_test_y.shape) # (200000, )

# print(df_train_x.info())
# print(df_train_y)
# print(df_test_x.info())
# print(df_test_y)
# print(df_res_id)

## x columns : bin_0 ~ 3, nom_0 ~ 9, ord_0 ~ 5, day, month
## y columns : target
## id fcolumns : id

##########################
### data preprocessing ###
##########################

## int64 / object 컬럼 분리
num_cols = [col for col in df_train_x.columns[:] if df_train[col].dtype in ['int64']]
cat_cols = [col for col in df_train_x.columns[:] if df_train[col].dtype in ['O']]


# print(num_cols)    # ['bin_0', 'bin_1', 'bin_2', 'ord_0', 'day', 'month']
# print(cat_cols)    # ['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

# # columns 정보 확인
for col in num_cols:
    # print("###", col, " ###")
    # print(df_train_x[col].head())
    # print(df_test_x[col].head())
    # print(df_train_x[col].describe())
    # print(df_test_x[col].describe())

    df_train_x[col] = df_train_x[col].astype('float64')
    df_test_x[col] = df_test_x[col].astype('float64')
    # print("#"*30, '\n')

###

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    # print("###", col, " ###")
    # print(df_train[col].describe())
    # print(df_test_x[col].describe())

    # print(df_train_x[col].head())
    # print(df_test_x[col].head())

    ## 문자형 data onehot encoding  수행
    le = LabelEncoder()
    le.fit(list(df_train_x[col].astype('str').values) +  list(df_test_x[col].astype('str').values))
    df_train_x[col] = le.fit_transform(df_train_x[col])
    df_test_x[col] = le.fit_transform(df_test_x[col])

    df_train_x[col] = df_train_x[col].astype('float64')
    df_test_x[col] = df_test_x[col].astype('float64')
    # print(df_train_x[col].head())
    # print(df_test_x[col].head())
    # print("#"*30, '\n')
 
train_x = df_train_x.values
test_x = df_test_x.values
train_y = df_train_y.values
test_y = df_test_y.values

print(train_x.shape)    # (300000, 23)
print(test_x.shape)     # (200000, 23)
print(train_y.shape)    # (300000, )
print(test_y.shape)     # (200000, )

### Scaler 적용

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()

scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/train_x.npy', train_x)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/test_x.npy', test_x)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/train_y.npy', train_y)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/test_y.npy', test_y)
'''