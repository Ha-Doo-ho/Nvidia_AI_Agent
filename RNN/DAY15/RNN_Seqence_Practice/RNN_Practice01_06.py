import tensorflow as tf
import numpy as np
import time 
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.metrics import AUC 

# 1-1 데이터
VOCABILARY= 10000
MAX_LEN = 200
EMBEDDING_DIM = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCABILARY)

# 1-2 데이터 전처리 
x_train = pad_sequences(sequences=x_train, maxlen=MAX_LEN, dtype="int32", padding="pre", truncating="pre")
x_test = pad_sequences(sequences=x_test, maxlen=MAX_LEN, dtype="int32", padding="pre", truncating="pre")

print(x_train.shape, y_train.shape) #(25000, 200) (25000,)
print(x_train) 
"""
[[   5   25  100 ...   19  178   32]
 [   0    0    0 ...   16  145   95]
 [   0    0    0 ...    7  129  113]
 ...
 [   0    0    0 ...    4 3586    2]
 [   0    0    0 ...   12    9   23]
 [   0    0    0 ...  204  131    9]]
"""

print(y_train)  #[1 0 0 ... 0 1 0]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=10, shuffle=True, stratify=y_train)

# 2 모델 구성
model = Sequential()
model.add(Input(shape=(200, )))
model.add(Embedding(input_dim=VOCABILARY, output_dim=EMBEDDING_DIM, mask_zero=True)) #input_dim: 사전에 등재된 단어 개수

model.add(LSTM(64, activation="tanh", return_state=False))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))
model.summary()

# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=10, mode="min", restore_best_weights=True)

start_time = time.time()
model.fit(x_train, y_train, batch_size=364, epochs=10, validation_data=(x_val, y_val), shuffle=True, callbacks=[es])
end_tiime = time.time()

real_time = np.round(end_tiime - start_time, 4)

# 4 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)
print(f"{y_predict} 소요시간: {real_time}")