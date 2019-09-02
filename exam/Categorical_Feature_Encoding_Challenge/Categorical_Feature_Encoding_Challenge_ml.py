import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

train_x = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/train_x.npy')
test_x = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/test_x.npy')
train_y = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/train_y.npy')

test_id = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_id.npy')


print(train_x.shape) # (300000, 23)
print(test_x.shape) # (200000, 23)
print(train_y.shape) # (300000, )
print(test_id.shape) # (200000, )



### model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.3
)

# rf = RandomForestClassifier(max_depth = 500, random_state=0, n_jobs=-1, n_estimators=1000)
# rf = RandomForestRegressor(max_depth = 500, random_state=0, n_jobs=-1, n_estimators=1000)

def build_model():
    # model = RandomForestClassifier(max_depth=600, n_jobs=-1, min_samples_leaf=10, n_estimators=1000, max_features='log2')
    model = RandomForestRegressor(max_depth=600, n_jobs=-1, min_samples_leaf=10, n_estimators=1000, max_features=None)
    return model


def create_hyperparameters():
    n_estimators = [100, 200, 400, 800, 1000]
    max_depth = [100, 200,300,400,500]
    min_samples_leaf = [1, 5, 10]
    max_leaf_nodes = [100, 1000, 2000]
    n_jobs = [-1]
    return {"n_estimators":n_estimators, "max_depth":max_depth, 
            "min_samples_leaf":min_samples_leaf, "max_leaf_nodes":max_leaf_nodes,
            "n_jobs":n_jobs}      # Map 형태로 반환

kfold = KFold(n_splits=5, shuffle=True)
# 학습하기
model = build_model()
# hyperparameters = create_hyperparameters()
# search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=15, n_jobs=1, cv=kfold)

model.fit(train_x, train_y)
test_score = model.score(val_x, val_y)
print("test_score :", test_score)

### fit and predict
y_pred = model.predict(test_x)

# df_test_y['target'] = pd.Series(y_pred, index=df_test_y.index)
df_res_data = {'id':test_id, 'target': y_pred}
df_res = pd.DataFrame(data=df_res_data, columns=['id', 'target'])

df_res.to_csv("submission.csv", mode='w', index=False)



'''
train_score : 0.8838441090411061
'''


'''
################
### csv read ###
################

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

# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

### model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# rf = RandomForestClassifier(max_depth = 500, random_state=0, n_jobs=-1, n_estimators=1000)
rf = RandomForestRegressor(max_depth = 500, random_state=0, n_jobs=-1, n_estimators=1000)
rf.fit(train_x, train_y)
train_score = rf.score(train_x, train_y)
print("train_score :", train_score)

### fit and predict
y_pred = rf.predict(test_x)

# df_test_y['target'] = pd.Series(y_pred, index=df_test_y.index)
df_res_data = {'id':df_test_y.values, 'target': y_pred}
df_res = pd.DataFrame(data=df_res_data, columns=['id', 'target'])

df_res.to_csv("submission.csv", mode='w', index=False)


'''