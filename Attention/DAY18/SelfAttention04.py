import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Embedding, Add, MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D  
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
PATH = "./_save/keras_Attention_imdb3.keras"

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)

# 1-2 데이터 전처리
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, padding="pre", truncating="pre")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=11, shuffle=True, stratify=y_train)
x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_val = pad_sequences(sequences=x_val, maxlen=MAXLEN, padding="pre", truncating="pre",)


# 2 모델 구성
inputs = Input(shape=(200, ))

TP_Embedding= TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(inputs)

attention_output = MultiHeadAttention(num_heads=3, key_dim=64)(query=TP_Embedding, key=TP_Embedding, value=TP_Embedding,)
# key_dim: Q,K, (value_dim을 사용하지 않으면 value_dim은 key_dim을 따라간다.) 의 차원이 얼마가 만들어지는지 결정한다. 지금 64니까 Q,K,V가 64차원이다.
# num_heads --> 동일한 문장을 "서로 다른 가중치를 가진" n개의 독립적인 Attention 계산 공간에서 분석함. 즉 WQ, WK, WV가 각각 3개씩 있다는 것이다. 지금은 3개다. GPT3는 96개임. 즉 64차원 Q, K, V가 각각 3개씩 나온다는 것이다. 더 문맥을 파악할 거리가 늘어난다는 것이다. 그만큼 무거워 지는 신호이기도 함.
"""
Head 1: W_Q¹, W_K¹, W_V¹
Head 2: W_Q², W_K², W_V²
Head 3: W_Q³, W_K³, W_V³
"""
# MultiHeadAttention 계층이 필요한 W_Q, W_K, W_V를 자동 생성하고 학습한다. 처음에는 초기화된 숫자이지만, 모델 훈련 과정에서 감정 분류에 유용한 Q·K·V가 나오도록 값이 수정된다.
# 이게 멀티 헤드 어텐션이라고 하는 것이다. num_heads를 1개가 아닌 n개로 설정할 수 있기 때문이다. 

x = GlobalAveragePooling1D()(attention_output,)

x = Dense(64, activation="relu")(x)

x = Dropout(0.3)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)

# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_loss", save_best_only=True, mode="min",)

start_time = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_val,y_val), callbacks=[es, mcp])
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

# 하나의 Attention_Layer에서는 괜찮다. 문제는 Layer가 깊어질수록 문제가 된다.
# 만약 AttentionLayer->MLP->AttentionLayer->MLP 순서대로 계속 학습한다면 즉 신경망이 깊어지면 학습 오차가 커지는 문제가 발생한다.
# 이는 역전파 과정에서 기울기가 소실되는 데, 깊어지면 깊어질 수록 더 심해진다. (이건 당연함) 즉, 기울기 소실 문제가 일어나 학습이 불안정 해진다. 
# 기울기 소실 문제를 해결하고 깊은 신경망의 학습을 안정화하 해야 한다.
# 그 방법이 원본(단어임베딩 + 포지션 임베딩)을 그대로 출력값에 더해주면 원본을 통해 학습 안정화가 가능하다. 