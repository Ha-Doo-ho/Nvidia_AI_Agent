# 1 기존 Dense AE 데이터 전처리 유지
# 2 VAE Encoder 구현
# 3 Sampling 계층 구현
# 4 Decoder 구현
# 5 VAE 손실 구현
 
import tensorflow as tf 
import numpy as np
import time 

from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# 1-1 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 1-2 데이터 전처리
# 이미지는 "초고차원 상의 한 점" 이라는 것을 무조건 가정하고 들어가야 함. --> 한 점 = 좌표로 표현 가능 = 좌표는 벡터로 표현함.
# 벡터 형태로 reshape 해야 한다. 기존 CNN은 위치 정보를 살리기 위해, -1, 28, 28, 1로 넣었다. 
""" 
입력 이미지 x
      ↓ Encoder
평균 μ와 표준편차 σ를 예측 (이게 모델이 예측하는 것이 분포의 파라미터라고 하는 이유이다.)
      ↓
그 분포에서 잠재벡터 z를 하나 샘플링
      ↓
Decoder에 z 전달

예를 들어 결과가 다음과 같다고 가정하자. 
μ = [2.0, -1.0]
σ = [0.2, 0.4]

이 운동화는 잠재공간에서 중심이 (2.0, -1.0)인 영역으로 표현되고,
첫 번째 축으로는 0.2 정도, 두 번째 축으로는 0.4 정도 퍼진 분포를 가진다.A

qΦ(z | x) = N( [ 2.0, -1.0 ], [ 0.2², 0.4² ] ) 로 표현되며 실제로는 여러 z가 나올 수 있다. 그 분포에서 뽑으므로 
"""
x_train = x_train.reshape(-1, 28 * 28 * 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28 * 1).astype("float32") / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=1, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(48000, 784) (10000, 784) (48000,) (10000,)

# 2 모델 구성
# encoder구성
LATENT_DIM = 2 #잠재벡터의 차원의 점. 현재는 2차원의 점
encoder_inputs = Input(shape=(28 * 28 * 1, )) #벡터의 형태로 들어감
encoder_x = Dense(units=128, activation="relu")(encoder_inputs)

#잠재벡터의 평균 --> 로그 분산과 마찬가지로 양,음수 가능해야 하므로 활성함수를 걸어 출력 범위를 제한하지 않았다.
z_mean = Dense(units=LATENT_DIM, name="z_mean")(encoder_x)

#잠재벡터의 log(분산) --> 분산은 음수가 될 수 없다. 일반적인 Dense출력은 음수가 나올 수 있다. log분산은 음수가 가능하다. 
z_log_var = Dense(units=LATENT_DIM, name="z_log_var")(encoder_x)

# Encoder는 이미지 1장을 받아서 평균 벡터, log 분산 벡터 2개를 출력한다.
encoder = Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var], name="encoder")
encoder.summary()
# 3 컴파일 및 훈련

# 4 평가 및 예측