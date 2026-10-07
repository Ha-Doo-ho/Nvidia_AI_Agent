# 1 기존 Dense AE 데이터 전처리 유지
# 2 VAE Encoder 구현
# 3 Sampling 계층 구현
# 4 Decoder 구현
# 5 VAE 손실 구현
 
 
# Sampling 계층을 구현해보자.
"""
잠재벡터를 중심으로 하여 표준편차만큼 퍼진 분포에서 뽑아본다. 그것이 이미지를 잘 생성(뽑을)할 확률이 높은 곳이기 때문이다. 

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
import tensorflow as tf 
import numpy as np
import time 

from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Layer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# 1-1 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 1-2 데이터 전처리
# 이미지는 "초고차원 상의 한 점" 이라는 것을 무조건 가정하고 들어가야 함. --> 한 점 = 좌표로 표현 가능 = 좌표는 벡터로 표현함.
# 벡터 형태로 reshape 해야 한다. 기존 CNN은 위치 정보를 살리기 위해, -1, 28, 28, 1로 넣었다. 
""" 
예를 들어 결과가 다음과 같다고 가정하자. 
μ = [2.0, -1.0]
σ = [0.2, 0.4]

이 운동화는 잠재공간에서 중심이 (2.0, -1.0)인 영역으로 표현되고,
첫 번째 축으로는 0.2 정도, 두 번째 축으로는 0.4 정도 퍼진 분포를 가진다.
"""
x_train = x_train.reshape(-1, 28 * 28 * 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28 * 1).astype("float32") / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=1, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(48000, 784) (10000, 784) (48000,) (10000,)

# 2 모델 구성
# encoder구성
LATENT_DIM = 2 #잠재벡터의 차원의 점. 현재는 2차원의 점

class Sampling(Layer):
    def call(self, inputs):
        
        # Encoder가 출력한 평균(μ)과 로그분산(logσ²)  --> input을 넣으면 평균과 분산(분포의 파라미터)이 나온다.
        # z_mean: Encoder가 이미지마다 만든 잠재분포의 평균 μ
        # z_log_var: Encoder가 이미지마다 만든 잠재분포의 로그분산logₑσ²
        # epsilon: 고정된 표준정규분포 N(0,1) 에서 추출.
        z_mean, z_log_var = inputs
        
        # 평균0 표준편차 1인 표준정규분포( tf.random.normal() 디폴트가 평균 0, 표편 1임. )에서 각 이미지마다 무작위 epsilon을 뽑는다.
        epsilon = tf.random.normal(shape=tf.shape(z_mean)) # z_mean이나 z_log_var의 차원은 어차피 같다. 둘 중 아무거나 넣는다. 잠재 벡터의 분포(차원)를 넣는다는 표현이 가장 정확함.
        
        # 표준편차를 구한다. 지수가 로그일 때 성질을 이용한다. 머신러닝에서는 log가 자연로그를 의미한다. logₑⁿ 를 의미한다는 것이다. eˡᴼᵍₑⁿ 의 성질로 n만 남기는 것이다. 
        z_std = tf.math.exp(0.5 * z_log_var) #분산(logσ²) 에 루트 씌운것이 표준편차이다. 이것을 자연상수 e(tf.math.exp)의 밑(base)으로 넣었다.  
        
        #Reparameterization 
        # 이미지를 무작위로 뽑기(샘플링) 때문에 미분이 불가능하다. 미분이 불가능 하다는 것은 역전파를 할 수 없다는 것과 같으므로 미분을 할 수 있도록 만든다.
        # 그 공식이  μ + σε (ε는 랜덤한 노이즈이다. --> epsilon) 이고, 이 값이 잠재 벡터가 된다.
        # 표준정규분포의 점 epsilon을 먼저 뽑은 다음 
        # z_std를 곱해서 퍼짐을 바꾸고,
        # z_mean을 더해서 중심을 이동한다. 
        """ 
        Encoder가 운동화 이미지 한 장을 보고 다음 값을 출력했다고 가정하자. 
        z_mean = [ 2.0, -1.0]
        z_std  = [ 0.2,  0.4]
        
        표준정규분포에서 우연히 다음 epsilon이 뽑혔다고 가정하자.
        epsilon = [0.5, -1.25]
        
        Reparameterization을 하면 최종 잠재 벡터는 다음과 같다. 
        z = [2.1, -1.5]
        
        다음번에 같은 이미지를 넣더라도 새로운 epsilon이 뽑히기 때문에 z가 조금 달라질 수 있다. 
        첫 번째 통과: z = [2.10, -1.50]
        두 번째 통과: z = [1.96, -0.83]
        세 번째 통과: z = [2.25, -1.12]

        하지만 모두 대체로 중심 [2.0, -1.0] 주변에서 뽑힌다. 
        """
        z = z_mean + z_std * epsilon 
        
        return z 

encoder_inputs = Input(shape=(28 * 28 * 1, )) # Dense층에 넣기 위해, 벡터의 형태로 들어감
encoder_x = Dense(units=128, activation="relu")(encoder_inputs)


#잠재벡터의 평균 --> 로그 분산과 마찬가지로 양,음수 가능해야 하므로 활성함수를 걸어 출력 범위를 제한하지 않았다.
z_mean = Dense(units=LATENT_DIM, name="z_mean")(encoder_x)

#잠재벡터의 log(분산) --> 분산은 음수가 될 수 없다. 일반적인 Dense출력은 음수가 나올 수 있다. log분산은 음수가 가능하다. 
z_log_var = Dense(units=LATENT_DIM, name="z_log_var")(encoder_x)

z = Sampling(name="sampling")([z_mean, z_log_var])
print("z의 shape" , z.shape)
# Encoder는 이미지 1장을 받아서 평균 벡터, log 분산 벡터 2개를 출력한다.
encoder = Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name="encoder")
print("z_mean.shape: ",z_mean.shape, "z_log_var.shape: ",z_log_var.shape)

encoder.summary()
#===========================================================
# 디코더 구현
#===========================================================

decoder_inputs = Input(shape=(LATENT_DIM, ), name="decoder_input")
decoder_x = Dense(units=128, activation="relu", name="decoder_hidden")(decoder_inputs)

# 128개의 특징을 원본 이미지 크기인
# 784개의 픽셀값으로 복원한다.
decoder_outputs = Dense(units=28 * 28 * 1, activation="sigmoid", name="decoder_output")(decoder_x) #마지막 출력층에 sigmoid를 사용하는 이유는 입력 이미지를 다음처럼 0~1로 정규화했기 때문이다. 이진분류를 위한 것이 아니다.

# Decoder을 독립된 모델로 구성
decoder = Model(inputs=decoder_inputs, outputs=decoder_outputs, name="decoder")

# Encoder의 Sampling 결과 z를 Decoder에 전달
reconstructed_outputs = decoder(z)

# Encoder와 Decoder가 연결된 전체 VAE 구조
# inputs는 인코더에 들어갈 것, 아웃풋은 decoder에 encoder을 통해 얻은 잠재벡터 z 이다.
VAE = Model(inputs=encoder_inputs, outputs = reconstructed_outputs, name="VAE")
VAE.summary()
# 3 컴파일 및 훈련

# 4 평가 및 예측