import pandas as pd
import numpy as np

################
### csv read ###
################

df_train = pd.read_csv("./data/kaggle/categorical_feature_encoding_challenge/train.csv")
df_test = pd.read_csv("./data/kaggle/categorical_feature_encoding_challenge/test.csv")


df_train_id = df_train['id']
df_test_id = df_test['id']
df_train_x = df_train.drop(['id', 'target'], axis=1)
df_train_y = df_train['target']
df_test_x = df_test.drop(['id'], axis=1)

num_cols = [col for col in df_train_x.columns[:] if df_train_x[col].dtype in ['int64']]
cat_cols = [col for col in df_train_x.columns[:] if df_train_x[col].dtype in ['O']]

np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_x_num.npy',
             df_train_x[num_cols].astype('int64').values)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_x_cat.npy',
             df_train_x[cat_cols].astype('str').values)

np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_x_num.npy',
             df_test_x[num_cols].astype('int64').values)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_x_cat.npy',
             df_test_x[cat_cols].astype('str').values)

np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_id.npy',
              df_train_id.astype('int64').values)
np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_id.npy',
             df_test_id.astype('int64').values)

np.save('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_train_y.npy',
              df_train_y.astype('int64').values)



