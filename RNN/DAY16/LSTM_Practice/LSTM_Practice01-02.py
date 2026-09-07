"""
LSTM, GRU 둘다 기존 RNN의 장기의존성 문제 해결을 위해 만들어졌다. 
LSTM과 GRU의 가장 결정적인 차이는 기억을 담는 공간(상태)의 개수와 정보를 통제하는 문(게이트)의 개수에 있다.

LSTM은 복잡한 장기 기억과 게이트를 통해 문맥을 섬세하게 파악한다.
GRU는 LSTM을 단순화한 구조로 연산효율을 극대화해 더욱 빠른 속도로 결과를 내는 것을 핵심으로 한다.


1. 기억공간의 차이 (2개 vs 1개)

LSTM (기억의 이원화): 장기 기억을 담당하는 셀 상태(ct)와 단기 기억(출력값)을 담당하는 은닉상태(ht)를 별도의 경로로 관리한다.
return_state=True 설정 시 모델이 ht, ct를 각각 (batch, n) 형태로 주는 것이다. (즉 장기기억과 단기기억 상태를 반환하겠냐는 것) 
장기기억 / 단기기억으로 이원화 되어 있다. 

GRU (기억의 통합): 장기기억을 담당하는 셀상태(ct)를 없애고, 모든 기억을 하나의 은닉 상태(ht)로 통합한다.
과거의 핵심 정보와 현재의 출력값이 하나의 컨베이어 벨트 위에서 함께 처리된다. 


2. 게이트 구조의 차이: 3개 vs 2개
LSTM (3개의 독립적인 문)
    1. 망각 게이트: 과거 기억 중 버릴 것을 결정한다.
    2. 입력 게이트: 현재 입력된 새 정보를 장기 기억에 얼마나 추가할지 결정한다.
    3. 출력 게이트: 업데이트된 장기 기억을 바탕으로 다음 시점으로 보낼 단기 기억을 출력한다. 
    
GRU (2개의 압축된 문)    
    1. 업데이트 게이트(Update): LSTM의 "망각"과 "입력" 역할을 하나로 합친 문이다. 
    "과거 기억을 얼만큼 지우고, 새 정보를 얼만큼 채워 넣을지"를 한 번의 연산 비율로 결정한다. 
    (예: 과거 정보 30% 유지, 새 정보 70% 반영)
    2. 리셋 게이트: 새 단어가 들어왔을 때 과거의 문맥을 얼마나 무시하고 초기화할지 결정한다. 
    
    
4. 실무적 이점과 보완점
결과적으로 GRU는 LSTM의 가장 큰 단점이었던 '과도한 파라미터(가중치)로 인한 느린 학습 속도와 높은 메모리 사용량'을 보완하기 위해 나왔다.
잡한 문을 통합하고 기억 공간을 하나로 줄였음에도 불구하고, 대다수의 자연어 처리 작업에서 LSTM과 거의 비슷한 정확도를 보여주기 때문에 실무에서 가성비 좋은 모델이라고 알려져 있다.

두 모델 모두 문장을 앞에서부터 뒤로 순차적으로 읽는 구조라는 공통점이 있다. 
이러한 단방향 구조의 한계를 극복하기 위해 문장을 양방향(앞→뒤, 뒤→앞)으로 동시에 읽어 문맥 파악 능력을 비약적으로 끌어올린 것이 Bidirectional LSTM이다. 

하지만 이렇게 해도 RNN 구조 자체의 고질적인 한계인 "문장이 너무 길어지면(수백 ~ 수천 단어) 결국 정보가 희미해진다" 는 문제는 완벽히 해결되지 않는다.
그래서 이 시계열 기반 처리 방식 자체를 완전히 폐기하고, 문장 내 단어들이 서로 어떤 관계를 맺고 있는지 '비율'로 한번에 계산하는 혁신적인 방법이 나왔다.  --> 그게 Attention이다.

그 Attention 구조를 핵심 기반으로 삼아 만들어진 신경망 모델이 Transformer(트랜스포머)이다. 
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
import matplotlib.pyplot as plt

# 1 - 1. 데이터
VOCAB = 10000
MAXLEN = 200
EMBEDING_DIM = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)
#print(x_train.shape, y_train.shape) #(25000,) (25000,)

print(len(x_train[0]),len(x_train[1]), len(x_train[2]), len(x_train[3]), len(x_train[4])) # 218 189 141 550 147

# 1 - 2 데이터 전처리 
# RNN 관련 모델은 모두 핵심이 같다. padding, truncating, masking
x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, padding="pre", truncating="pre")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=330, stratify=y_train)


# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(200, )))
model.add(Embedding(input_dim=VOCAB, output_dim=EMBEDING_DIM, mask_zero=True))
model.add(Bidirectional(LSTM(64, activation="tanh", return_sequences=False)))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(1, activation="sigmoid"))s
model.summary()


# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])
es = EarlyStopping(monitor="val_loss", patience=10, mode="min", restore_best_weights=True, )

start_time = time.time()
model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1, callbacks=[es], validation_data=(x_val, y_val))
end_time = time.time()

real_time = np.round(end_time - start_time , 4)


# 4 예측 및 평가 
loss, accuracy, auc = model.evaluate(x_test, y_test, batch_size=128)
y_predict = model.predict(x_test, batch_size=128)
y_predict_class = (y_predict > 0.5).astype("int32")

print(f"loss: {loss}, accuracy: {accuracy}, 소요시간:{real_time}초")
print(y_predict_class)

cm_labels = ["Negative", "Positive"]
cm = confusion_matrix(y_test, y_predict_class)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
cm_display.plot(cmap="Oranges", xticks_rotation=45)
