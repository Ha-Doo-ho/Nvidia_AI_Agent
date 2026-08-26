import tensorflow as tf 
import time 
import pandas as pd
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, RandomRotation,RandomFlip, Conv2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Dense
from keras.regularizers  import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt 

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1) reshape가 필요 없다. 원래 채널 수가 빠져 있을 때만 사용한다. 

x_train = x_train.reshape(-1, 32, 32, 3).astype("float32") / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype("float32") / 255.0

# 2. 모델 구성 
cnn_model = Sequential()
cnn_model.add(Input(shape=(32, 32, 3)))
#cnn_model.add(RandomRotation(factor=0.08, fill_mode="nearest", interpolation="bilinear", seed=30))
cnn_model.add(RandomFlip(mode="horizontal",seed=34))

cnn_model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2))

cnn_model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2))

cnn_model.add(Conv2D(filters=128, kernel_size=3, padding="valid", activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2))

cnn_model.add(GlobalAveragePooling2D())
cnn_model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
cnn_model.add(Dense(10, activation="softmax"))

# 3. 컴파일 및 훈련
cnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

es = EarlyStopping(monitor="val_loss",patience=10, mode="min", restore_best_weights=True)
start_time = time.time()
cnn_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[es])
end_time = time.time()

# 4. 평가 및 예측
loss, accuracy = cnn_model.evaluate(x_test, y_test)
y_predict = cnn_model.predict(x_test)

y_predict_class = np.argmax(y_predict, axis=1)

cm = confusion_matrix(y_test, y_predict_class)
cm_display_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_display_labels)
cm_display.plot(cmap="Blues_r", xticks_rotation=45)
plt.show()

print(f"accuracy: {np.round(accuracy,4)}, 소요시간:{np.round(end_time - start_time, 4)}")