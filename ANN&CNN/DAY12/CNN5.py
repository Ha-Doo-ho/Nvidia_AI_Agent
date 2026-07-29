import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist

# 🌟 L2 규제를 사용하기 위해 regularizers를 임포트합니다.
from keras import regularizers 

# ==========================================
# 1. 데이터 (생략 - 이전과 동일)
# ==========================================
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


# ==========================================
# 2. 모델 구성 (L2 규제 적용)
# ==========================================
cnn_model = Sequential(name="cnn_with_l2")
cnn_model.add(Input(shape=(28, 28, 1), name="Input_layer"))

# --- 1차 특징 추출기 ---
cnn_model.add(Conv2D(32, kernel_size=3, padding="same", activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2)) 

# --- 2차 특징 추출기 ---
# 💡 [여기!] L2 규제를 Conv2D에 적용해 봅니다. 
# 1e-4는 0.0001을 뜻하며, 규제 강도(lambda)를 나타냅니다. 작게 시작하는 것이 좋습니다.
cnn_model.add(Conv2D(64, kernel_size=3, padding="same", activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2))

# --- 통역사 ---
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.25)) 

# --- 뇌세포 (은닉층) ---
# 💡 [여기!] 파라미터가 가장 많은 Dense 층에 L2 규제를 적용하면 효과가 좋습니다.
cnn_model.add(Dense(64, activation='relu', 
                    kernel_regularizer=regularizers.l2(1e-4))) 
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5)) 

# --- 출력층 ---
cnn_model.add(Dense(10, activation="softmax")) 


# ==========================================
# 3. 컴파일 및 훈련 / 4. 평가 및 예측 (이전과 동일하게 진행)
# ==========================================