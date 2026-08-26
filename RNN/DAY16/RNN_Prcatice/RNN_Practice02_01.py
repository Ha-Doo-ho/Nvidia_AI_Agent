import tensorflow as tf
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Input,TextVectorization,Embedding,LSTM,Dense,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================================================
# 1. 데이터
# =========================================================

x_train = np.array([
    "this movie is very good",
    "I really enjoyed this movie",
    "the acting was wonderful",
    "this movie is very bad",
    "I hated this film",
    "the story was boring"
])

# 1: 긍정, 0: 부정
y_train = np.array([1, 1, 1, 0, 0, 0])


# =========================================================
# 2. 주요 설정
# =========================================================

MAX_TOKENS = 10_000       # Vocabulary의 최대 토큰 수 --> 사전에 등재할 수 있는 단어의 종류
MAX_LENGTH = 100          # 문장 하나의 최대 토큰 길이 --> 임베딩 되기 전, 문장을 id로 구분할 때의 개수이다!!
EMBEDDING_DIM = 128       # 토큰 하나를 표현할 실수 특징 수 --> 이때는 벡터로 바꿨을 때 그 개수이다!! 구분 잘 해야 함. 


# =========================================================
# 3. TextVectorization
# =========================================================

vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAX_LENGTH
)

# 반드시 훈련 데이터로 Vocabulary를 먼저 생성
vectorizer.adapt(x_train)

vocab_size = vectorizer.vocabulary_size()

print("실제 Vocabulary 크기:", vocab_size)


# =========================================================
# 4. 모델 구성
# =========================================================

model = Sequential(name="text_lstm_model")

# 문자열 한 문장을 하나의 데이터로 입력
model.add(Input(shape=(),dtype=tf.string,name="text_input"))

# 문자열 → 정수 토큰 ID
model.add(vectorizer)

# 정수 토큰 ID → 학습 가능한 실수 벡터
model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True,name="embedding"))

# 문장 순서와 문맥 학습
model.add(LSTM(units=64,name="lstm"))
# 과적합 방지
model.add(Dropout(rate=0.3,name="dropout"))

# 이진 분류
model.add(Dense(units=1,activation="sigmoid",name="output"))


# =========================================================
# 5. 컴파일
# =========================================================

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()


# =========================================================
# 6. 콜백
# =========================================================

early_stopping = EarlyStopping(monitor="val_loss",mode="min",patience=5,restore_best_weights=True)

model_checkpoint = ModelCheckpoint(filepath="./text_lstm_best.keras", monitor="val_loss", mode="min", save_best_only=True)


# =========================================================
# 7. 학습
# =========================================================

start_time = time.time()
history = model.fit(x_train,y_train,epochs=30,batch_size=2,validation_split=0.2,callbacks=[early_stopping, model_checkpoint],verbose=1)
end_time = time.time()

print(f"학습 시간: {end_time - start_time:.2f}초")


# =========================================================
# 8. 예측
# =========================================================

x_test = np.array([
    "this movie was wonderful",
    "this movie was boring"
])

y_probability = model.predict(x_test)

# sigmoid 결과가 0.5 이상이면 긍정
y_predict = (y_probability >= 0.5).astype("int32")

for text, probability, prediction in zip(
    x_test,
    y_probability.flatten(),
    y_predict.flatten()
):
    label = "긍정" if prediction == 1 else "부정"

    print(f"문장: {text}")
    print(f"긍정 확률: {probability:.4f}")
    print(f"예측 결과: {label}")
    print()