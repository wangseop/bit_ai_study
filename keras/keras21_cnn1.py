from keras.models import Sequential

filter_size = 7
kernel_size = (2,2)
model = Sequential()

from keras.layers import Conv2D ,MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(filter_size, kernel_size, padding='valid',         # padding default는 valid, same으로 주면 input_shape 크기를 유지
                input_shape = (5,5,1)))     # (5,5,1) 이미지를 (2,2)로 잘라서 4x4로 만들어내고 filter_size 장 수 만큼 결과 가져온다 
                                            # => output Shape = (None, 4,4,7) 출력됨
                                            # (4,4) 크기의 이미지가 7장 나온다
model.add(Conv2D(16, (2,2)))                # (3,3) 으로 자르면 한 이미지 당 2x2만큼의 특징점 추출
model.add(Conv2D(8, (2,2)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())                        # 
model.add(Dense(10))

model.summary()
