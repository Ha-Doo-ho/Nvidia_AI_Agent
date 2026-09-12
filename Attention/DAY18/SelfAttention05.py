# 이번 단계에서는 Attention 앞의 입력을 Attention 결과와 다시 더한다.
"""
기존 진행상황

토큰 ID
→ 토큰+위치 임베딩
→ Multi-Head Self-Attention
→ GlobalAveragePooling1D
→ 감정 분류 

수식으로 표현하면
X = TokenAndPositionEmbedding(inputs)
A = MultiHeadAttention(num_heads, Q/K/V차원)(Q,K,V)

그 다음 바로
문장벡터 = GlobalAveragePooling1D(A)를 수행한다.

여기서 알 수 있는 것.
X: 원래 단어 의미 + 위치 정보
A: 다른 단어와의 관계를 반영한 Attention 결과 

=========================================================================

이제 진행하는 것.
                ┌──────────────────┐
                │                  ↓
Embedding → Attention → Add → LayerNormalization
                ↑          ↑
                └ Residual ┘
                번역하면 잔차 라고 한다.
                
                
이전 구조의 한계
X가 Attention을 통과한 결과인 A만 다음 계층으로 전달된다. X는 전달되지 않는다.
문맥을 잘 반영은 하는데, Attention이전의 값인 원본이 없어서 원래 단어 의미 유지가 살짝 어려워질 수 있다.
물론 Attention을 통과한 A도 원래 단어 정보가 들어 있을 수 있다. 그러나 다른 문맥 정보를 추가하는 것에 조금 더 초점이 맞추어져 있다.

Residual Connection은 원래 입력을 보존하는 직접 경로를 만들어 Attention의 부담을 줄인다. --> 이것이 목표이다.

X
↓
MultiHeadAttention
↓
A
↓
GlobalAveragePooling1D

Attention에서는 원래 임베딩을 다음과 같이 여러 차례 변환한다.
128차원 임베딩
→ Q, K, V로 변환
→ Q와 K 관계 점수 계산
→ V들을 가중합
→ 출력 가중치로 다시 128차원 변환
따라서 Attention 결과에는 문맥 정보가 추가되지만, 원래 입력 X가 그대로 전달되는 통로는 없다.

이전 모델도 감정 분류 모델로 정상적으로 학습을 할 수 있다. 
문제는 Attention층을 여러개 쌓아 Transformer Encoder로 발전시킬 때,  다음 2가지 문제가 발생할 수 있다.

1. 원래 단어·위치 정보를 다음 층으로 전달하는 직접 경로가 없다. 
2. 여러 층을 거치면서 학습 신호가 이전 층까지 전달되기 어려워질 수 있다.

Residual Connection(잔차 연결 --> 딥러닝에서 레이어의 입력값을 출력값에 직접 더해주는 --> ResNet에서 본 게 이것임.)
Rgₒₒd = Xgₒₒd + Agₒₒd 

다음과 같이 가정하자. 
Xgₒₒd (단어 임베딩 + 포지션 임베딩) = [0.8,−0.4,0.6,0.2]

Agₒₒd (어텐션 통과 후) = [−0.1,0.7,0.2,−0.3]

두 벡터를 더한다. Rgₒₒd = [0.8,−0.4,0.6,0.2] + [−0.1,0.7,0.2,−0.3]
코드로는 다음과 같다. x = Add()([TP_Embedding,attention_output]) 혹은 x = TP_EMbedding + attention_output

ResNet의 핵심 아이디어인 각 계층, 대리-> 과장 -> 이사 . . . 사장 을 거치는 것이 아닌 사장으로 한번에 갈 수 있는 경로를 제공하는 것. '
이 직접 경로가 핵심이다.  이 덕분에 모델이 깊어질수록, 원본을 참조 하지 못하는 경우를 최대한 줄일 수 있다.

거의 대부분 잔차 연결을 한다고 보면 된다. 딥러닝의 가장 큰 문제인 기울기 소실 문제를 해결하는 가장 보편이고, 효과적인 방법이다. 
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

# 1-1 데이터
VOCAB = 10000
MAXLEN = 200
EMBEDDING_DIM = 128
PATH = "./_save/keras_Attention_imdb3.keras"

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, dtype="int32", padding="pre", truncating="pre")

# 1-2 데이터 전처리
# 패딩이 여기서 구현된다. 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=11, shuffle=True, stratify=y_train, train_size=0.9)
x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, dtype="int32", padding="pre", truncating="pre")
x_val = pad_sequences(sequences=x_val, dtype="int32", padding="pre", truncating="pre", maxlen=MAXLEN,)

# 2 모델 구성
inputs = Input(shape=(MAXLEN, ))

# MaskZero로 마스킹처리를 할 수 있다. 마스킹 한것은 PositionEmbedding으로 반영될 수는 있다. 그러나 읽으면 안된다.  
# 단어 임베딩 + 포지션 임베딩 = E 
TP_embedding = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(inputs)

# E를 Wᵠ, Wᴷ, Wⱽ 와 각각 행렬 곱셈하여 Q, K, V를 제작한다. 그때 Q,K 몇차원, V 몇차원으로 할 것인지 결정이 가능하다. 그런데, 보통 임베딩한 차원의 절반으로 설정한다. 
# Q, K를 내적하여 가장 큰 값을 가진 단어가 내가 원하는 단어를 잘 주목(Attention)하고 있다는 의미이다. 여기서 Softmax를 걸어서(알기 쉽게 해야 하므로) 다음 나올 단어를 파악하거나, 문맥을 알아낸다. --> 3Blue1Brown보기
# MultiHeadAttention은 그러한 W₁ᵠ. . . Wₙᵠ | W₁ᴷ. . .Wₙᴷ | W₁ⱽ. . . Wₙⱽ 환경을 제작해서 더 문맥을 깊이 파악할 수 있도록 돕는다. 
attention_output = MultiHeadAttention(num_heads=3, key_dim=64)(query=TP_embedding, key=TP_embedding, value=TP_embedding)

#Attention 후에 바로 Dropout 적용도 가능하다.
attention_output = Dropout(0.3, name="attention_dropout")(attention_output)

# 추가된 부분1 --> Residual Connection(잔차 연결 --> 딥러닝에서 레이어의 입력값을 출력값에 직접 더해주는 --> ResNet에서 본 게 이것임.) 
# Rgₒₒd = Xgₒₒd + Agₒₒd 이다. 원본을 계속 넣어줌으로써 기울기 소실 문제를 
# 기울기 소실 문제를 해결하고 깊은 신경망의 학습을 안정화하기 위해 사용한다.
# x = TP_embedding + attention_output 해도 된다. 어차피 행렬 덧셈이다. 
# 이전에는 수정본만 다음 사람에게 전달되고 원본은 전달 안되었지만, 이제는 수정내용 + 원본을 전달할 수 있게 되었다.
x = Add(name="attention_residual")([TP_embedding, attention_output]) 

# 추가된 부분2
# LayerNormalization은 “각 토큰 벡터의 숫자들을 일정한 기준으로 조정” 한다. 더 정확하게 말한다면, 128개의 값을 평균 0, 표편 1에 가깝게 표준화한다.
# 만약 4차원이라고 한다면 다음과 같이 나올 수 있다. [2,4,6,8] ⟶ [−1.34,−0.45,0.45,1.34] 음수가 나왔다고 문제되진 않는다. 합계가 1이될 필요도 없다. 어차피 출력단에서 activation function 걸어줄 것이다. 
x = LayerNormalization(axis=-1, epsilon=1e-6, name="attention_layer_norm")(x)

# GlobalAveragePooling은 “여러 토큰 벡터를 하나의 문장 벡터로 요약” 한다. CNN, RNN과 마찬가지로 필수 계층은 아니다. 
"""
토큰 자리	특징 1	특징 2	특징 3	특징 4
movie	      2	    4	    6	    8
not	          8	    6	    4	    2
good	      2	    6	    2	    6

GlobalAveragePooling1D()는 같은 특징 번호끼리 토큰 전체에 걸쳐 평균을 낸다.
특징 1 평균: (2+8+2) / 3 = 4

나머지도 계산하면: 문장 대표 벡터=[4,5.33,4,5.33] 4차원 벡터 3개를, 4차원 벡터 1개로 요약하였다. 그래서 필수가 아니다. 속도적인 측면에서 고려된다. 
"""
x = GlobalAveragePooling1D(name="global_average_pooling")(x)

# 감정 분류층
x = Dense(64, activation="relu", name="classifier_dense")(x)
x = Dropout(0.2, name="classifier_dropout")(x)
outputs = Dense(1, activation="sigmoid", name="sentiment_output")(x)
model = Model(inputs=inputs, outputs=outputs)

# 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_accuracy", mode="max", save_best_only=True)

start = time.time()
model.fit(x_train, y_train, batch_size=64, epochs=7, callbacks=[es, mcp], validation_data=(x_val, y_val), shuffle=True)
end = time.time()
real = np.round(end-start, 4)

# 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test).flatten()
print(f"loss:{loss}, accuracy:{accuracy}, auc:{auc}")
print("소요시간: ",real)
print(y_predict)
