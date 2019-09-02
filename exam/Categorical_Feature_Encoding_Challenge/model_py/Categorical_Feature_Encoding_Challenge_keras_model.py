from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

model = Sequential()
# model.add(Dense(500, input_shape=(23, ), activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation='relu'))

# model.add(Dense(700, activation='relu'))
# model.add(Dense(700, activation='relu'))
# model.add(Dense(700, activation='relu'))

# model.add(Dense(900, activation='relu'))
# model.add(Dense(900, activation='relu'))
# model.add(Dense(900, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(400, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.4))

# model.add(Dense(600, activation='relu'))
# model.add(Dense(600, activation='relu'))
# model.add(Dense(600, activation='relu'))
# model.add(Dropout(0.4))

model.add(Dense(1000, input_shape=(23, ), activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='sigmoid'))

model.add(Dense(1500, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(1500, activation='relu'))

model.add(Dense(2500, activation='relu'))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(600, activation='relu'))
model.add(Dense(600, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=['mse', 'acc'])

model.summary()

model.save('./exam/Categorical_Feature_Encoding_Challenge/model_file/keras_model.h5')