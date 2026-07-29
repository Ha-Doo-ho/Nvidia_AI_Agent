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

"""
실무와 학계의 표준에서는 일반적인 Dropout을 보통 Dense 층(또는 Flatten 직후)에만 집중적으로 사용합니다.

이미지의 공간적 연관성 (가장 핵심적인 이유)
이미지를 보면 픽셀 하나하나가 독립적이지 않고, 바로 옆에 있는 픽셀들과 색상, 모양이 거의 같다는 특징을 가진다. 
GPT 다크모드 사용하면 검은 픽셀 주변은 대부분 검은 픽셀인 것이 그 이유이다. 

Conv2D를 통과한 데이터에 일반 Dropout을 걸어 특정 픽셀(노드)을 무작위로 꺼버렸다고 가정해 보겠습니다.
모델의 꼼수: 모델은 주변 픽셀의 값을 이용해서 꺼진 값을 유추할 수 있다. 그래서 DropOut이 그렇게 효과를 보지는 못한다. (좋아질 여지는 있으나 미미하다는 것)

그래서 Conv층에서 Dropout을 걸고 싶다면, 일반 Dropout 대신 SpatialDropout2D가 더 적합할 수 있다
일반 Dropout을 사용하면 특징 맵의 개별 위치를 무작위로 제거한다. 
그래서 주변 픽셀을 보고 유추할 수 있기 때문에, 일반 Dropout이 효과가 미미할 것이라고 하는 것이다. 
SpatialDropout2D는 feature 맵 한 채널 전체를 제거한다. 
Keras공식 문서에서는 Conv 층에서는 일반 Dropout보다 SpatialDropout2D를 사용하도록 권유한다.  
"""