import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist # Fashion-MNIST 데이터셋 로드
from keras.layers import GlobalAveragePooling2D

# ==========================================
# 1. 데이터
# ==========================================
# Keras에서 제공하는 훈련용/테스트용 데이터를 바로 나누어 불러옵니다.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)
# 이것을 보면 채널이 없다는 것을 알 수 있다. 그래서 채널을 추가하기 위해 reshape로 채널을 추가하는 것이다. 

# [중요] CNN은 채널(두께) 정보가 반드시 필요합니다!
# 현재 x_train은 (60000, 28, 28) 형태이므로, 끝에 1(흑백)을 추가해 (60000, 28, 28, 1)로 모양을 바꿔줍니다.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 이미지 스케일링 (정규화)
# 픽셀 값은 0~255 사이입니다. 이를 0~1 사이로 나누어주면 모델이 훨씬 빠르고 정확하게 학습합니다.
x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0

print("x_train shape:", x_train.shape) # (60000, 28, 28, 1) 확인
print("y_train shape:", y_train.shape) # (60000,) 확인

#print(y_train.features)


# ==========================================
# 2. 모델 구성 (질문자님이 작성하신 뼈대 그대로!)
# ==========================================
cnn_model = Sequential(name="cnn_baseline")
cnn_model.add(Input(shape=(28, 28, 1), name="Input_layer"))

# 1차 특징 추출기
cnn_model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation='relu', name="conv_1"))
cnn_model.add(MaxPool2D(pool_size=2, name="pool_1")) # 주석에 쓰신 대로 기본 strides가 pool_size를 따라갑니다! 패딩도 안하는게 맞습니다.
# filters가 32개면 바이어스도 32개. 그리고 입력 채널의 개수를 뜻한다. 

# 2차 특징 추출기
cnn_model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation='relu', name="conv_2"))  
cnn_model.add(MaxPool2D(pool_size=2, name="pool_2"))
#filters가 64개이므로 바이어스도 64개

# 통역사 (2차원 -> 1차원)
cnn_model.add(Flatten(name="flatten"))

# 피처들 형태가 다 다를 때

# 뇌세포 (최종 판단)
cnn_model.add(Dense(64, activation='relu', name="hidden")) 
cnn_model.add(Dense(10, activation="softmax", name="output")) 


# ==========================================
# 3. 컴파일 및 훈련
# ==========================================
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# DNN에서 쓰셨던 EarlyStopping 적용! (CNN은 연산이 무거워 patience를 10~20 정도로만 줘도 충분합니다)
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)

# 훈련 시작 (validation_split으로 검증 데이터를 자체 분리합니다)
cnn_model.fit(
    x_train, y_train, 
    epochs=10, 
    batch_size=128, # CNN은 연산량이 많아 batch_size를 조금 넉넉히(32, 64, 128 등) 주는 것이 좋습니다.
    validation_split=0.2, 
    callbacks=[es],
)


# ==========================================
# 4. 평가 및 예측
# ==========================================
# evaluate는 loss와 컴파일할 때 넣었던 metrics(여기서는 accuracy)를 리턴합니다.
loss, accuracy = cnn_model.evaluate(x_test, y_test)
print(f"최종 Test Loss: {loss:.4f}")
print(f"최종 Test Accuracy: {accuracy:.4f}")

# 예측
y_predict = cnn_model.predict(x_test)

# y_predict는 10개 클래스에 대한 '확률'로 나옵니다. (softmax의 결과)
# np.argmax를 쓰면 그중 가장 확률이 높은(가장 정답이라고 확신하는) 인덱스(클래스 번호)만 쏙 뽑아줍니다.
y_predict_classes = np.argmax(y_predict, axis=1)

print("실제 정답 10개: ", y_test[:10])
print("모델 예측 10개: ", y_predict_classes[:10])

cnn_model.summary()