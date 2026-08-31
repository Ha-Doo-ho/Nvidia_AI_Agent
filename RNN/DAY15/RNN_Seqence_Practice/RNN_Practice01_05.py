import tensorflow as tf
import numpy as np
import time 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Embedding, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC

# 발전된 형태는 GlobalAveragePooling1D() --> SimpleRNN을 사용한다는 것이다.

VOCAB_SIZE = 10000      # 사전에 들어갈 단어의 개수 
MAX_LENGTH = 200        # 한 문장의 길이  
EMBEDDINT_DIM = 128     # 문장은 단어로 이루어져 있는데, 그 단어를 벡터로 만든다. 그 벡터를 몇차원으로 할 것인가? 크면 더 다양한 단어를 담을 수 있을 것 


PATH = "./_save/keras_RNN_imdb_SimpleRNN.keras"

# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE) #상위 10000개의 단어를 가져온다. 그게 사전이 된다. 
print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(x_train[0]) #[1, 14, 22, 16, 43, 530, 973, 1622, 1385 . . . ., 113, 103, 32, 15, 16, 5345, 19, 178, 32]
print(y_train) #[1 0 0 ... 0 1 0] 2만 5000개 있음. 답지이다. 0은 Negative한 반응, 1은 Positive한 반응


# 1-1 데이터 전처리 (RNN 문장 분석은 Padding, Truncating, Masking을 생각해야 함)
x_train = pad_sequences(sequences=x_train, maxlen=MAX_LENGTH, padding="pre", truncating="post",)
x_test = pad_sequences(sequences=x_test, maxlen=MAX_LENGTH, padding="pre", truncating="pre", )

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=11, stratify=y_train)

print("x_train:", x_train.shape) #(17500, 200) 17500개의 문장, 각 문장은 200 length이다.
print("x_val:", x_val.shape)     #(7500, 200)   7500개의 문장, 각 문장은 200 length이다.  
print("x_test:", x_test.shape)   #(25000, 200) 25000개의 문장, 각 문장은 200 length이다.

# 2. 모델 구성
# 한 문장을 넣는데, 그게 긍정인지 부정인지 알고 싶은 것이다. Input은 200개의 단어 가 된다. 
model = Sequential()
model.add(Input(shape=(200, )))
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDINT_DIM, mask_zero=True)) # 단어를 넣으면 임베딩해야 한다. input_dim은 사전에 등재된 단어의 개수, output_dim은 하나의 단어를 표현할 때 몇 차원의 Vector로 표현하는지를 의미한다. 

model.add(SimpleRNN(64, activation="tanh", return_sequences=False)) #은닉층
# 200개의 단어 임베딩을 순서대로 처리한다.
# 각 시점에서 이전 은닉 상태와 현재 단어 임베딩을 함께 계산한다.
# 마지막 유효 단어까지 처리한 은닉 상태만 출력한다.
# shape: (batch, 200, 128) → (batch, 64)


model.add(Dense(64, activation="relu")) # 은닉층 중에서 분류층
model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid")) #출력층
model.summary()

# 3. 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_accuracy", save_best_only=True, mode="max",)

start_time = time.time()
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), shuffle=True)
end_time = time.time()

real_time = np.round(end_time-start_time, 4)

# 4. 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test)
print(f"loss:{loss}, accuracy:{accuracy}, auc:{auc}")

# y_predict = model.predict(x_test)
# print(f"{y_predict}")
"""
[[2.5992449e-02]
 [9.9445373e-01]
 [7.2185969e-01]
 ...
 [4.0994948e-04]
 [4.5511657e-05]
 [1.2096620e-04]]
 
 2차원 형태이다. 1차원으로 바꾸던가 차원 변경을 해야 한다. 그래서 기본 쌩짜로 찍어보라는 것이다. 
"""

y_predict = model.predict(x_test).flatten()
y_predict_probablity = (y_predict > 0.5).astype("int32")
print(f"{y_predict_probablity}")
print(f"소요시간: {real_time}")

