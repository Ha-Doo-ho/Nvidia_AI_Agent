import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt 
import time

#AutoEncoder

PATH = "./_save/keras_autoencoder01.keras"

# 1 데이터

# Autoencoder는 답지(y)가 클래스가 아니다. 답지(y)가 자기 자신이다.
# y_train, y_test를 받아서 사용하지 않는다.
(x_train, _), (x_test, _) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

# 1-2 데이터 전처리 (정규화 및 train - val - test로 분리) --> 픽셀값은 0~255까지 존재한다.
# Conv2D와 Dense가 요구하는 입력 형태의 차이
# 이미지 1장을 입력으로 받는다. Conv2D와 Dense가 요구하는 입력 형태의 차이가 있다. 
# CNN은 픽셀의 ※공간적 배치※를 이용해야 하므로 (높이, 너비, 채널) 형태를 유지한다. 
# Convolution autoencoder라면 CNN처럼 입력하면 된다. 
# 벡터 안에는 모든 흑백 픽셀값이 들어 있다. [픽셀1, 픽셀2, 픽셀3, ..., 픽셀784]
x_train = x_train.reshape(-1, 28 * 28 * 1).astype("float32") / 255.0 #컬러라면 -1, 28 * 28 * 3 이다.
x_test = x_test.reshape(-1, 28 * 28 * 1).astype("float32") / 255.0

x_train, x_val = train_test_split(x_train, train_size=0.9, random_state=11, shuffle=True)
print(x_train.shape, x_val.shape) # (54000, 784) (6000, 784)



# 2 모델 구성
INPUT_DIM = 28 * 28 * 1  # 흑백 이미지 한 장의 전체 픽셀 수: 784
LATENT_DIM =  32         # 이미지를 압축해서 표현할 잠재 벡터 크기

autoencoder = Sequential()

# Encoder
# 784개의 픽셀값을 128개의 중간 특징으로 변환
autoencoder.add(Input(shape=(INPUT_DIM, )))
"""
h = ReLU(xW + b)가 동작한다. 
x: 784개 픽셀
W: 학습되는 가중치
b: 학습되는 편향
h: 128개의 중간 특징

128개의 값은 원본 픽셀을 단순히 128개 선택한 것이 아니다. 784개 픽셀을 서로 다른 비율로 조합하여 만든 학습된 특징이다. 
"""

autoencoder.add(Dense(units=128, activation="relu", name="encoder_hidden"))

# 128개의 특징을 32개의 잠재 특징으로 압축
# 이 계층의 출력이 잠재 벡터이며 Bottleneck이다.
autoencoder.add(Dense(units=LATENT_DIM, activation="relu",name="latent_vector"))

# Decoder
# 32개의 잠재 특징을 다시 128개의 특징으로 확장 
# ※잠재 벡터 32개 = 전체 이미지 정보를 조합한 32개의 학습된 특징
autoencoder.add(Dense(units=128, activation="relu", name="decoderhidden"))

# 128개의 특징을 원래 픽셀 개수인 784개로 복원
# 정답 픽셀값이 0~1이므로 sigmoid를 사용한다.  → softmax는 0~1의 확률 값이다. 다 더하면 1이다. 이건 확률값 구할 때(다중분류) 사용하는 것이다. 그래서 부적절 한 것이다.  모든 픽셀을 다 더하니 1이다? 이게 아니므로.
autoencoder.add(Dense(units=INPUT_DIM, activation="sigmoid", name="reconstruction_output"))
autoencoder.summary()

""" 
중요한 점: Decoder는 Encoder의 계산을 단순히 역순으로 되돌리는 장치가 아니다.
Encoder의 가중치 ≠ Decoder의 가중치 --> 가중치가 서로 다르다.

두 부분은 각각 별도의 가중치를 학습한다.
    · Encoder: 복원에 필요한 정보를 32차원에 담는 방법 학습
    · Decoder: 32차원 정보로 784개 픽셀을 만드는 방법 학습 (잠재 벡터를 이미지로 복원함.)
    
    w
"""


# 3 컴파일 및 훈련
autoencoder.compile(optimizer="adam", loss="mse", metrics=["mse"])

es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_loss", save_best_only=True, mode="min")

start = time.time()
autoencoder.fit(x=x_train, y=x_train, batch_size=64, epochs=10, validation_data=(x_val, x_val), shuffle=True, callbacks=[es, mcp]) # y가 모델이 복원해야 할 정답 이미지. 이므로 그대로 들어간다.
end = time.time()

real_time = np.round(end-start, 4)
print("소요시간: ", real_time)

# 평가 및 예측
loss, mse = autoencoder.evaluate(x_test, x_test)
x_reconstructed = autoencoder.predict(x_test)

print(f"loss:{loss}, mse:{mse}") #loss:0.01134648360311985, mse:0.01134648360311985
print(x_reconstructed) #[[2.0110463e-07 1.1378940e-07 6.2045137e-07 ... 1.7009065e-06 . . .  [9.3637682e-06 1.4022233e-05 2.2635282e-05 ... 3.0932797e-04 9.4387797e-05 3.6331541e-05]]

IMAGE_COUNT = 10
for i in range(IMAGE_COUNT):

    # 첫 번째 줄: 원본 이미지
    plt.subplot(2, IMAGE_COUNT, i + 1)

    original_image = x_test[i].reshape(28, 28)

    # Matplotlib이 픽셀값을 어떤 밝기로 표시할지 정하는 기준
    # cmap="gray": 작은 값은 검게, 큰 값은 희게 표시
    # vmin=0: 0을 가장 검은색으로 표시
    # vmax=1: 1을 가장 흰색으로 표시
    plt.imshow(original_image,cmap="gray", vmin=0, vmax=1) 

    plt.title("Original")
    plt.axis("off")


    # 두 번째 줄: Autoencoder가 복원한 이미지
    plt.subplot(2, IMAGE_COUNT, IMAGE_COUNT + i + 1)

    reconstructed_image = x_reconstructed[i].reshape(28, 28)

    plt.imshow(reconstructed_image,cmap="gray",vmin=0,vmax=1) 

    plt.title("Reconstructed")
    plt.axis("off")


plt.show()


""" 
학습은 다음 과정의 반복이다.
원본 이미지
→ Encoder
→ n차원 잠재 벡터
→ Decoder
→ 복원 이미지
→ 원본과 비교
→ 복원 오차 계산 (여기서 mse 사용됨)
→ 역전파
→ Encoder와 Decoder 가중치 수정


다음 역할을 함께 학습하게 될 것이다.
Encoder: 어떤 정보를 32개 값에 남겨야 하는가?
Decoder: 그 32개 값으로 어떻게 이미지를 복원해야 하는가?
--> 원본 데이터를 더 작고 의미 있는 내부 표현으로 바꾸는 것.


어디서 사용하나? 

1. 이상 탐지
Autoencoder의 대표적인 활용 분야

우선 정상 데이터만 사용하여 학습한다.
정상 제품 이미지
→ Autoencoder 학습


학습이 끝나면 다음을 알 수 있게 된다.

정상 제품
→ 익숙한 형태
→ 잘 복원됨
→ 복원 오차가 작음

불량 제품
→ 학습하지 않은 형태
→ 잘 복원되지 않음
→ 복원 오차가 큼

예를 들어 공장에서 정상 나사 이미지만 학습했다고 가정하자.
정상 나사: MSE = 0.005
금이 간 나사: MSE = 0.080

복원 오차가 임계값 보다 크면 불량으로 판단할 수 있다.
 
활용 예시는 다음과 같다..

제조업 불량 탐지
의료영상 이상 탐지
서버·센서 이상 상태 탐지
금융 거래 이상 탐지
네트워크 침입 탐지


2. 노이즈 제거
Denoising Autoencoder는 손상된 데이터를 입력하고 깨끗한 데이터를 정답으로 사용한다.

노이즈 이미지
→ Encoder
→ 중요한 구조 보존
→ Decoder
→ 깨끗한 이미지 복원

예를 들어 다음과 같이 학습한다. → model.fit(x=noisy_images, y=clean_images)

활용 분야는 다음과 같다.

오래된 사진의 노이즈 제거
저조도 영상 개선
의료영상 잡음 감소
센서 데이터 잡음 제거

"""