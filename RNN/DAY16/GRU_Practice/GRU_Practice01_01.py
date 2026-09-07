import tensorflow as tf 
import pandas as pd
import time 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Embedding, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping


# 1. 데이터
# 사전, 최대 길이, 임베딩 시 몇차원 벡터?
VOCAB = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB) 

# 1-1 데이터 전처리
x_train = pad_sequences(sequences=x_train, maxlen=MAX_LENGTH, padding="pre", truncating="pre")
x_test = pad_sequences(sequences=x_test, maxlen=MAX_LENGTH, padding="pre", truncating="pre")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=11, shuffle=True, stratify=y_train)

# 2 모델 구성
model = Sequential()
model.add(Input(shape=(200, )))
model.add(Embedding(input_dim=VOCAB, output_dim=EMBEDDING_DIM, mask_zero=True)) #input_dim: 사전에 등재된 단어 개수

model.add(GRU(64, activation="tanh", return_state=False))

"""

return_sequences:	모든 단어 위치의 은닉 상태를 출력할 것인가?
return_state:	    마지막 상태 h_T, C_T를 별도로 추가 반환할 것인가?    

| `return_sequences` | `return_state` | 반환 결과                                     
| ------------------ | -------------- | --------------------------------------       
| `False`            | `False`        | 마지막 출력 `(batch,64)`                      
| `True`             | `False`        | 모든 시점 출력 `(batch,200,64)`               
| `False`            | `True`         | `output`, `h_T`, `C_T` 각각 `(batch,64)`      
| `True`             | `True`         | 전체 시퀀스 `(batch,200,64)`와 `h_T`, `C_T`    

"""

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))
model.summary()

# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=10, mode="min", restore_best_weights=True)

start_time = time.time()
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val), shuffle=True, callbacks=[es])
end_tiime = time.time()

real_time = np.round(end_tiime - start_time, 4)

# 4 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test) 

# 결과가 어떻게 나올 지 모른다. 그래서 predict를 바로 출력해서 어떠한 형태로 나오는지를 확인을 해보아야 한다. 물론 Sigmoid라서 확률값(실수) 으로 나올 것은 알고 있으나, 차원 등 다른 문제가 있을 수 있다. 
# y_predict = model.predict(x_test)
# print(f"{y_predict} 소요시간: {real_time}")

"""
[[0.23496214]
 [0.969997  ]
 [0.4596889 ]
 ...
 [0.04264813]
 [0.06709462]
 [0.5217781 ]] 소요시간: 107.8114
 
 2차원 형태의 "실수"가 나온다. sigmoid를 통해 0초과 1미만의 수로 나오는데, 이는 긍정인지 부정인지를 가르는 것이다. 0.5 이하면 부정으로 생각할 수 있고, 0.5 초과면 긍정으로 볼 수 있다. 
 첫번째 문장은 23%이므로 부정에 가깝고 세번째는 긍정 96%이므로 긍정에 가깝다.    
"""

y_predict = model.predict(x_test).flatten()
y_predict_class = (y_predict > 0.5).astype("int32")
print(y_predict, "소요시간: ",real_time)


cm = confusion_matrix(y_test, y_predict_class)
labels = ["Negative", "Positive"]
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(cmap="plasma", xticks_rotation=45)
plt.show()
