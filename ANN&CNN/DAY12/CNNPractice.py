import tensorflow as tf
import numpy as np 
from keras.models import Sequential
from keras.layers import Dropout, RandomRotation, BatchNormalization, Flatten, GlobalAveragePooling2D,Conv2D, Input, Dense, MaxPool2D
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 1 데이터 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # 자동으로 train, test 셋이 분리되어 있음. 
print(x_train.shape) #(60000, 28, 28)
print(x_test.shape)  #(10000, 28, 28)

# 무조건 인풋레이어 행|열|채널
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2 모델 구성
cnn_model = Sequential() #NN 을 여기서 만듦
cnn_model.add(Input(shape=(28,28,1))) #데이터 개수 무시 행/열/차원우선 암기할 것. 
cnn_model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding = "same", activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(2))

cnn_model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(2))

cnn_model.add(Dense(64, activation="relu"))
cnn_model.add(Dense(10, activation="softmax"))

# 3 컴파일 및 훈련
cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy'])
cnn_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2, )

# 4 평가 및 예측
