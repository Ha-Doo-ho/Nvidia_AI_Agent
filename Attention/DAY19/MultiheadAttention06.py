""" 
FFN이 추가되었다. 이미 배운 잔차 연결과 LayerNormalization을 FFN 뒤에도 적용하면 Transformer Encorder 블록 1개가 된다.

1. Attention + 잔차 연결 까지 했는데, 왜 FFN 이 필요한가? 
현재는 Attention이 다른 토큰의 정보를 가져와 각 토큰의 표현을 바꾸고 있다.

ex)
    this movie is not good 
에서 good 위치의 벡터에 not과 관련된 정보가 반영될 수 있다. --> Attention에서 이미 not을 참고하여 good벡터에 넣을 수 있으며, LayerNorm은 그 실행된 것을 정규화 한다. 
이제 그 벡터 안에는 여러 정보가 함께 담겨 있다. FFN은 이 특징들을 조합하여, 다음 처리에 유용한 새로운 특징을 만드는 역할을 한다. 


구성	    수행하는 일
-------------------------------------------------------
Attention	토큰 사이의 관계를 이용해 정보를 섞음
FFN	        각 토큰 안의 특징들을 비선형적으로 재조합함
-------------------------------------------------------
예를 들어 모델은 “긍정적인 표현과 그것을 부정하는 문맥이 함께 나타나는 경우”에 반응하는 특징을 학습할 수 있다. 
물론 실제 벡터의 특정 번호가 반드시 <긍정/부정>을 담당하는 것은 아니다. 

현재 모델에는 Pooling뒤에 Dense가 있지만, ★그때는 이미 토큰들이 하나의 리뷰 벡터로 합쳐진 상태★이다. 
--> “토큰별 정보를 각각 가공한 뒤 합칠 것인가, 먼저 합친 정보를 가공할 것인가"의 차이다. (FFN을 적용하기 전에는 후자를 따랐다.)

★FFN을 Pooling 앞에 넣으면 각 토큰의 문맥 표현을 추가로 가공한 뒤 리뷰 전체를 요약할 수 있다.★ --> LayerNorm과 함께 더 정확한 문맥파악을 위해 사용된다. ★
핵심은 Attention이고 보조 수단으로 LayerNormalization, FFN이 사용된다. 

Attention 자체에도 Softmax라는 비선형 연산이 존재하듯이 FFN도 별도의 비선형 특징 변환 능력을 추가하는 것이다. 
기본 Transformer의 FFN은 이미 배운 Dense 두 개와 그 사이의 활성화 함수로 구성된다. 논문에서는 ReLU를 사용한다. 

"""

import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Embedding, Add, MultiHeadAttention,LayerNormalization, Dense, Dropout, GlobalAveragePooling1D
from keras_hub.layers import PositionEmbedding, TokenAndPositionEmbedding
from keras import ops
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.metrics import AUC
import numpy as np
import time 
import matplotlib.pyplot as plt 

# 1 데이터
VOCAB = 10000
MAXLEN = 200
EMBEDDING_DIM = 128
PATH = "./_save/keras_Attention_imdb4.keras"

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)

# 1-2 데이터 전처리
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, padding="pre", truncating="pre")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=11, shuffle=True, stratify=y_train)
x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_val = pad_sequences(sequences=x_val, maxlen=MAXLEN, padding="pre", truncating="pre",)


# 2. 모델 구성
FF_DIM = 256

inputs = Input(shape=(MAXLEN,), dtype="int32")

TP_Embedding = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(inputs)

# ── 앞서 배운 Attention 부분 ──

attention_output = MultiHeadAttention(num_heads=3, key_dim=64)(query=TP_Embedding, key=TP_Embedding, value=TP_Embedding)

attention_output = Dropout(0.1)(attention_output)

# 첫 번째 잔차 연결과 LayerNormalization
x = Add()([TP_Embedding, attention_output])
x = LayerNormalization(axis=-1, epsilon=1e-6)(x)
# x: (batch, 200, 128)


# ── 이번에 추가하는 FFN 부분 ──
# 기본 Transformer의 FFN은 Dese 2개와 그 사이의 활성화 함수로 구성된다. 두번째에 relu 를 넣지 않는 이유는 2번째 Dense의 가중치가 음수일 수도 있는데, relu를 적용하면, 특징을 줄이는 변화가 없어질 수 있기 때문이다.
ffn_output = Dense(FF_DIM, activation="relu")(x) #(16, 200, 128)⟶(16, 200, 256)  입력 특징들을 256가지 가중합으로 계산하고 ReLU 적용
# (batch, 200, 256)

"""
어떤 토큰의 벡터가 다음과 같다고 가정. 
--> x=[2, 3] 이라고 가정

이 두 숫자를 이용하여 서로 다른 계산 3개를 해보자. 편향은 영향이 크지 않으니 0으로 설정하였음.

출력            계산에 쓰는 가중치          계산                결과
첫 번째 출력        [1, 1]              2 x 1 + 3 x 1           5
두 번째 출력        [-1, 1]             2 x (-1) + 3 x 1        1
세 번째 출력        [1, -2]             2 x 1 + 3 x (-2)       -4 (음수다. 도움이 되지는 않는 정보일 확률이 있다. 아닐 수도 있음.)

입력은 두 개였지만, 계산을 세 종류 했으므로 결과는 세 개. 
여기에 activation="relu"를 적용하면 음수가 0으로 바뀐다. -->  [5, 1, -4]⟶[5, 1, 0] --> 이걸 어려운 말로 비선형 변환 이라고 부른다. 

===============================================================================

| 항목                          변경 전 | 변경 후 
| 리뷰 개수                      |   16 |   16 
| 리뷰마다 토큰 자리 수           |  200 |  200 
| 토큰 하나를 표현하는 숫자 개수   |  128 |  256 

모든 토큰에 "같은 가중치 행렬(파라미터)" 을 적용하지만, 들어오는 벡터가 다르므로 계산 결과는 달라질 수 있다. 
"""

ffn_output = Dense(EMBEDDING_DIM)(ffn_output) # ★그 결과를 다시 조합★해 128개 숫자로 표현
# (batch, 200, 128)

ffn_output = Dropout(0.1)(ffn_output)

# 두 번째 잔차 연결: FFN 입력 + FFN 결과
x = Add()([x, ffn_output])

x = LayerNormalization(axis=-1, epsilon=1e-6)(x)
# 여기까지가 Encoder 블록 하나
# x: (batch, 200, 128)
"""
일부러 출력 숫자를 더 많이 만드는가? 

FFN에서 서로 다른 특징 조합을 계산할 자리를 더 많이 마련하기 위해서 이다. 
앞의 예시에서는 같은 [2, 3]으로 다음처럼 다른 계산을 했다. 
    · 두 값을 더하는 계산.
    · 첫 번째 값을 빼고 두 번째 값을 더하는 계산.
    · 두 번째 값에 더 큰 음수 가중치를 주는 계산. 

--> 실제 FFN에서는 어떤 조합이 감정 분류에 도움이 되는지를 학습한다. 중간 출력이 256개이면, 이러한 계산을 256개 마련할 수 있다. 
그리고 ReLU와 다음 Dense 계층을 함께 사용해서, 이 계산 결과들을 다시 조합한다. --> 84, 111 Line 

"""

# ── 기존 리뷰 요약과 감정 분류 부분 ──

x = GlobalAveragePooling1D()(x)
# (batch, 128)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()


# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_loss", save_best_only=True, mode="min",)

start_time = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val,y_val), callbacks=[es, mcp])
end_time = time.time()

real_time = np.round(end_time-start_time, 4)

# 4 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict(x_test).flatten()
y_predict_class = (y_predict >= 0.5).astype("int32")
print(f"정확도: {round(accuracy, 4)}, 소요시간: {real_time}")

cm = confusion_matrix(y_test, y_predict_class)
labels = ["Negative", "Positive"]

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap="Oranges_r", xticks_rotation=45)
plt.show()