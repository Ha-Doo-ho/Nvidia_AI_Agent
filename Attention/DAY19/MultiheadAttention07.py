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
FF_DIM = 256 #피드 포워드 디멘션
PATH = "./_save/keras_Attention_imdb5.keras"
(x_train, y_train), (x_test, y_test)= imdb.load_data(num_words=VOCAB)

# 1-2 데이터 전처리 
pad_sequences(sequences=x_test, maxlen=MAXLEN, dtype="int32", padding="pre", truncating="pre")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=11, shuffle=True, stratify=y_train)

x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_val = pad_sequences(sequences=x_val, dtype="int32" , maxlen=MAXLEN, padding="pre", truncating="pre")

# 2 모델 구성
inputs = Input(shape=(MAXLEN, ))

# 단어 / 포지션 임베딩 
TP_EMBEDDING = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM,mask_zero=True)(inputs)

# MultiheadAttention을 구성함
# num_heads가 3이므로  W₁ᵠ. . . Wₙᵠ | W₁ᴷ. . .Wₙᴷ | W₁ⱽ. . . Wₙⱽ 환경인데, n이 3일 것이고, 각 가중치는 Q,K,V를 64차원으로 만들 가중치 행렬을 제작할 것이다. 
attention_output = MultiHeadAttention(num_heads=3, key_dim=64)(query=TP_EMBEDDING, key=TP_EMBEDDING, value=TP_EMBEDDING)

# 잔차 연결 (Residual Connection) 어텐션에서 여러 단어들에 문맥이 고려된 정보가 이미 포함되었을 수 있다. 이때, 층이 깊어지면 원본의 내용이 변질될 위험이 있다. 
# ResNet의 핵심 아이디어인 각 계층, 대리-> 과장 -> 이사 . . . 사장 을 거치는 것이 아닌 사장으로 한번에 갈 수 있는 경로를 제공하는 것. 
# 이 직접 경로가 핵심이다.  이 덕분에 모델이 깊어질수록, 원본을 참조 하지 못하는 경우를 최대한 줄일 수 있다. 
# 기울기 소실 문제를 해결하고 깊은 신경망의 학습을 안정화하기 위해 사용한다.
x = Add()([TP_EMBEDDING, attention_output]) # 혹은 그냥 x = TP_EMBEDDING + attention_output 해도 된다. 

# LayerNormalization
# LayerNormalization은 “각 토큰 벡터의 숫자들을 일정한 기준으로 조정” 한다. 더 정확하게 말한다면, 128개의 값을 평균 0, 표편 1에 가깝게 표준화한다.
# CNN의 BatchNormalization과 동일한 역할을 한다고 보면 된다. 잔차 연결과 함께, 학습을 안정화 하고 학습 가속화를 위해 사용한다. 물론 정규화는 늘 원본의 내용을 깎아낸다는 단점이 있다.
x = LayerNormalization(epsilon=1e-5)(x)

# Feed Forword
# 단어간 관계 파악은 Attention에서 상당 부분 완료된다. 여기서는  단어 간 관계 파악을 넘어 각 토큰(단어)의 표현(차원)을 정교하게 비선형 변환하는 데, 사용한다.
# 그 방법이 차원을 늘려서 활성화 함수 적용하고 다시 원래대로 되돌리는 것이다.
# 역시 Attention all you need에 들어가 있다. 
ffn_output = Dense(units=FF_DIM, activation="relu")(x)
ffn_output = Dense(units=EMBEDDING_DIM)(ffn_output)
ffn_output = Dropout(0.2)(ffn_output)

# 2번째 Residual_Connection 및 LayerNormalization
x = x + ffn_output
x = LayerNormalization(axis=-1, epsilon=1e-4)(x)

x = GlobalAveragePooling1D()(x)
x = Dense(units=64, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(units=1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)

# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=10, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_loss", save_best_only=True, mode="min",)

start_time = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=30, validation_data=(x_val,y_val), callbacks=[es, mcp])
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