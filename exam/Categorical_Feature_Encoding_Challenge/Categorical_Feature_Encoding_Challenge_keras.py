import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

train_x = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/train_x.npy')
test_x = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/test_x.npy')
train_y = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/train_y.npy')

test_id = np.load('./data/kaggle/categorical_feature_encoding_challenge/npy/ori_test_id.npy')


print(train_x.shape) # (300000, 23)
print(test_x.shape) # (200000, 23)
print(train_y.shape) # (300000, )
print(test_id.shape) # (200000, )

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.3
)

### model
from keras.models import load_model
model = load_model('./exam/Categorical_Feature_Encoding_Challenge/model_file/keras_model.h5')

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='acc', patience=30)

history = model.fit(train_x, train_y, epochs=300, batch_size=9192, 
                    validation_data=(val_x, val_y), callbacks=[es])

print("Train Accuracy :", np.mean(history.history['acc']))
print("Validation Accuracy :", np.mean(history.history['val_acc']))

### fit and predict
y_pred = model.predict(test_x)


# df_test_y['target'] = pd.Series(y_pred, index=df_test_y.index)
df_res_data = {'id':test_id.reshape(-1), 'target': y_pred.reshape(-1)}
df_res = pd.DataFrame(data=df_res_data, columns=['id', 'target'])

df_res.to_csv("submission.csv", mode='w', index=False)





    