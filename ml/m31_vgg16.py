from keras.applications import VGG16 ,VGG19, Xception, InceptionV3, ResNet50, MobileNet
from keras.layers import Dense, Flatten
from keras.models import Model, Sequential

Xception(), InceptionV3(), ResNet50(), MobileNet()
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
#conv_base = VGG16()   # default input_shape=(224,224,3) , default include_top=True
"""
방법 1.
x = conv_base.output
x = Flatten()(x)
x = Dense(256 ,activation="relu")(x)
x = Dense(1, activation='sigmoid')(x)
conv_base = Model(conv_base.input, x)

방법 2.
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu))
model.add(layers.Dense(1, activation='sigmoid'))
"""

model = Sequential()
model.add(conv_base)
model.add(Flatten())        # conv_base가 이미 Flatten() 상태였다면 사용시 Error 생길 수 있다
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
