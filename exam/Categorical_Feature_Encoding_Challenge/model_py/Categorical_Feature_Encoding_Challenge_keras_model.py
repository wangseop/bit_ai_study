from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(500, input_shape=(23, ), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))

model.add(Dense(700, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(700, activation='relu'))

model.add(Dense(900, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(900, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.save('./exam/Categorical_Feature_Encoding_Challenge/model_file/keras_model.h5')