import tensorflow as tf
import numpy as np
import time 
from keras.models import Sequential
from keras import regularizers
from keras.layers import Input, RandomRotation, Conv2D,MaxPool2D, BatchNormalization, Flatten, GlobalAveragePooling2D, Dense
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt 


# 1 Data 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape) # (60000, 28, 28)

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)


# 2 Model Configuration 
cnn_model = Sequential(name="cnn_baseline")
cnn_model.add(Input(shape=(28,28,1)))  # 데이터 개수 무시, 행|열|차원 우선 , Input_layer

cnn_model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation='relu'))
cnn_model.add(BatchNormalization()) #BatchNormalization으로 규제걸기 
cnn_model.add(MaxPool2D(2))

cnn_model.add(Conv2D(filters=64, kernel_size=3, padding="same",))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(2))

#cnn_model.add(Flatten()) 이것도 사용할 수 있다. 이것은 1차원으로 쫙 펼친다. 
cnn_model.add(GlobalAveragePooling2D()) #각 feature_map에서 나온 평균을 쭉 잇는다. 
cnn_model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
cnn_model.add(Dense(10, activation='softmax'))

# L2 규제를 사용하는 주된 목적은 모델이 훈련 데이터에 너무 과하게 맞춰지는 과적합(Overfitting)을 막기 위해서이다. 
# 어차피 인공지능 연산은 곱연산이다. 정규화를 하고, 어떤 개지랄을 해도 숫자 커지면 높은 가중치 걸리는 순간, 그 값으로 오도된다. 그걸 막는 장치이다. (추가하고 안좋으면 빼거나 10배씩 늘림)

# 3. Compile and fit
cnn_model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor="val_loss", patience=100, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint("./_save/keras_CNN1.keras", monitor="val_loss", save_best_only=True, )

start_time = time.time()
cnn_model.fit(x_train, y_train, batch_size=64, epochs=10, callbacks=[es, mcp], validation_split=0.2, )
end_time = time.time()

# 4. Model Evalutate and Predict
loss, accuracy = cnn_model.evaluate(x_test, y_test) 
y_predict = cnn_model.predict(x_test)
y_predict_class = np.argmax(y_predict)


