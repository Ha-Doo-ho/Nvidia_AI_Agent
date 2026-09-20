import tensorflow as tf
import numpy as np
import time
from csv import QUOTE_NONE
import pandas as pd 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold, KFold
import re # 정규표현식 지원
from keras.layers import Input, TextVectorization, MultiHeadAttention, Add, LayerNormalization, Dropout, Dense, Softmax
from keras_hub.layers import TokenAndPositionEmbedding
import string
from keras.models import Model
from keras.metrics import AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1-1 데이터
NUM_SAMPLES = 50000 # 문장 몇개 사용할 것 인지.
MAXLEN = 40 # 의미가 2개 존재한다. 1.영어의 최대 문장, 2. [start] 부터 독일어 문장 끝까지 ([end]는 포함하지 않음.)
VOCAB = 10000 # 단어 사전

PATH = "./_data/deu.txt"

# 1-2 데이터 전처리
# pandas의 read_csv 기능으로 txt 파일이 바로 데이터프레임으로 만들 수 있다. 
# ※ pandas에서 데이터 프레임 내부의 객체가 문자열일 경우, 내부의 객체에 접근할 때, 반드시 str로 우선 접근하고, 메서드를 붙여야 한다. 
data_csv = pd.read_csv(filepath_or_buffer=PATH,
                   sep="\t",
                   header=None, # 파일에 열 이름이 없을 때 사용
                   names=["English", "German", "Attribution"],
                   dtype=str,
                   encoding="utf-8",
                   keep_default_na=False, #"NA" 같은 문장을 결측값으로 바꾸지 않음
                   quoting=QUOTE_NONE) #문장 속 따옴표를 CSV 제어문자가 아닌 일반 문자로 처리

data_csv = data_csv.drop(labels=["Attribution"], axis=1)

print(data_csv.shape) #(331266, 2)
print(data_csv.columns) #Index(['English', 'German'], dtype='str')

# 문자열 앞 / 뒤 공백 제거
# "   I am tired.   " --> "I am tired." 기존 내부의 띄어쓰기는 제거 안하므로 안심할 것.
data_csv["English"] = data_csv["English"].str.strip() 
data_csv["German"] = data_csv["German"].str.strip()

# 빈문장 제거
data_csv = data_csv[
    (data_csv["English"] != "") & (data_csv["German"] != "")
]

# 중복 문장 쌍 제거 문장이 중복해서 들어갔으면 그건 필요 없으니까 지운다. 
data_csv = data_csv.drop_duplicates(subset=["English", "German"])

# 전체 데이터에서 무작위로 50,000개 선택
# .sample()메서드는 DataFrame에서 행을 무작위로 선택한다. 0, 1, 2, 3, 4, 5 있으면 n=3일때 3, 0, 4 이렇게 가져올 수 있음.
# 이게 인덱스에 그대로 적용되서 문제이다. 3, 0, 4를 그대로 사용할 수 없다. 다시 0, 1, 2로 바꿔야 한다. 그래서 .reset_index를 사용하는 것이다. 거의 같이 쓴다고 보면 됨.
data_csv = data_csv.sample(n=min(NUM_SAMPLES, len(data_csv)), random_state=11).reset_index(drop=True)

# ※핵심
# 영어 --> 독일어. 영어를 독일어로 번역한다. 번역 당하는 Target Language 에서 sos, eos사용함. (번역하는 언어는 Source Language)
# [start] Ich bin müde. [end] 모든 독일어 문장은 이렇게 된다. 
data_csv["German"] = "[start] " + data_csv["German"] + " [end]"

# x: Encoder 입력. y: Decorder용 독일어 문장 
x = data_csv["English"]
y = data_csv["German"] 

# 홀드아웃 방식으로 데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=11, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.5, random_state=11, shuffle=True)

print(data_csv.shape)                # (50000, 2)
print(x_train.shape, y_train.shape)  # (28000,) (28000,)
print(x_val.shape, y_val.shape)      # (12000,) (12000,)
print(x_test.shape, y_test.shape)    # (10000,) (10000,)

#print(data_csv[["English", "German"]].head())

# 1-3 TextVectorization
# 다음 shifted right 단계에서 41개를 입력 40개와 정답 40개로 나누기 위해 사용 
# --> 디코더 입력: [start] 부터 끝 단어 / 정답: 시작 단어부터 [end]까지
GERMAN_MAXLEN = MAXLEN + 1 # --> MAXLEN은 영어 문장의 최대 길이이자, start가 포함된 독일어 문장이다. 
                           #     GERMAN_MAXLEN은 end도 포함되어 있어서 +1이다.

# 독일어 문장 표준화 함수 시작
# 일반적인 문장부호 전부 제거하는 함수: punctuation
# 그러나 [start], [end]의 대괄호는 보존해야 한다.
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[","")
strip_chars = strip_chars.replace("]","")
strip_chars = re.escape(strip_chars) # 특수 문자를 escape할 수 있게 도와줌. 문법으로 읽지 않게 하는 것이다.
#print(data_csv[["English", "German"]].head())

def german_standardization(input_text):
    lowercase_text = tf.strings.lower(input_text)
    return tf.strings.regex_replace(
        lowercase_text,
        f"[{strip_chars}]",
        ""
    )

# 영어 TextVectorization
english_vectorizer = TextVectorization(max_tokens=VOCAB, standardize="lower_and_strip_punctuation", split="whitespace", output_mode="int", output_sequence_length=MAXLEN, name="English_vectorizer")

#독일어 TextVectorization
german_vectorizer = TextVectorization(max_tokens=VOCAB, standardize=german_standardization, split="whitespace", output_mode="int", output_sequence_length=GERMAN_MAXLEN)

# --------------------------------------------------
# 훈련 데이터로 어휘 사전 생성
# --------------------------------------------------

# validation과 test 데이터는 adapt에 사용하지 않는다.
english_vectorizer.adapt(x_train.to_numpy())
german_vectorizer.adapt(y_train.to_numpy())


# 만들어진 실제 어휘 사전
english_vocabulary = english_vectorizer.get_vocabulary()
german_vocabulary = german_vectorizer.get_vocabulary()

ENGLISH_VOCAB_SIZE = len(english_vocabulary)
GERMAN_VOCAB_SIZE = len(german_vocabulary)

print("영어 실제 어휘 수:", ENGLISH_VOCAB_SIZE)
print("독일어 실제 어휘 수:", GERMAN_VOCAB_SIZE)

# 1-4 Outputs shifted right
# 영어 문장 → Encoder 입력 토큰
# 독일어 문장 → 41칸짜리 전체 토큰 
encoder_train_tokens = english_vectorizer(x_train.to_numpy())
encoder_val_tokens = english_vectorizer(x_val.to_numpy())
encoder_test_tokens = english_vectorizer(x_test.to_numpy())

german_train_tokens = german_vectorizer(y_train.to_numpy())
german_val_tokens = german_vectorizer(y_val.to_numpy())
german_test_tokens = german_vectorizer(y_test.to_numpy())
print(german_train_tokens.shape) #(20000, 41)

# 여기까지의 관계
""" 
x_train: 영어 문자열
→ english_vectorizer
→ encoder_train_tokens

y_train": [start] 독일어 문장 [end]
→ german_vectorizer
→ german_train_tokens
"""

# 41개 전체 Target 토큰을 입력 40개와 정답 40개로 나눈다.
decoder_train_inputs = german_train_tokens[:, :-1]
decoder_train_labels = german_train_tokens[:, 1:]

decoder_val_inputs = german_val_tokens[:, :-1]
decoder_val_labels = german_val_tokens[:, 1:]

decoder_test_inputs = german_test_tokens[:, :-1]
decoder_test_labels = german_test_tokens[:, 1:]

# PAD 토큰 0은 번역 정답이 아니므로 손실과 정확도에서 제외한다.
train_sample_weights = tf.cast(
    decoder_train_labels != 0,
    dtype=tf.float32,
)

val_sample_weights = tf.cast(
    decoder_val_labels != 0,
    dtype=tf.float32,
)

test_sample_weights = tf.cast(
    decoder_test_labels != 0,
    dtype=tf.float32,
)


# 2 모델 구성
EMBEDDING_DIM = 128
NUM_HEADS = 4
KEY_DIM = EMBEDDING_DIM // NUM_HEADS #32임
FF_DIM = 512

# 임베딩(문자 + 위치) 뒤, Dropout
encoder_inputs = Input(shape=(MAXLEN, ), dtype="int32", name="encoder_inputs")
encoder_embedding = TokenAndPositionEmbedding(vocabulary_size=ENGLISH_VOCAB_SIZE, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True, name="encoder_embedding")(encoder_inputs)
encoder_x = Dropout(rate=0.1)(encoder_embedding)

# Encoder Self_attention --> Query, Key, Value가 모두 자기 자신
attention_output1 = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(query=encoder_x, key=encoder_x, value=encoder_x)

# Add & Norm (여기서는 LayerNorm을 말함)
x1 = Add()([encoder_x, attention_output1])
x1 = LayerNormalization(axis=-1, epsilon=1e-5, name="encoder1_attention_norm")(x1)

# FeedForward
ffn_forward = Dense(units=FF_DIM, activation="relu")(x1)
ffn_forward = Dense(units=EMBEDDING_DIM)(ffn_forward)
ffn_forward = Dropout(rate=0.1)(ffn_forward)

# Add & Norm
encoder1_output = Add()([x1, ffn_forward])

# 2번째 encoder 제작
x2 = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(query=encoder1_output, key=encoder1_output, value=encoder1_output)

# Add & Norm
x2 = Add()([encoder1_output, x2])
x2 = LayerNormalization(axis=-1, epsilon=1e-5)(x2)

# FeedForward
ffn_forward = Dense(units=FF_DIM, activation="relu")(x2)
ffn_forward = Dense(units=EMBEDDING_DIM)(ffn_forward)
ffn_forward = Dropout(rate=0.1)(ffn_forward)

# Add & Norm
x2 = Add()([x2, ffn_forward])
encoder2_output = LayerNormalization(axis=-1, epsilon=1e-5)(x2)


# Decoder input & 단어,Positional_Encoding
decoder_inputs = Input(shape=(MAXLEN, ), dtype="int32", name="decoder_input")
decoder_embedding = TokenAndPositionEmbedding(vocabulary_size=GERMAN_VOCAB_SIZE, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(decoder_inputs)

# Masked Multi-Head Attention
dx1 = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(query=decoder_embedding, key=decoder_embedding, value=decoder_embedding, use_causal_mask=True)

# Add & Norm
dx1 = Add()([dx1, decoder_embedding])

# Cross-Multi-Head Attention
dx2 = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(query=dx1, key=encoder2_output, value=encoder2_output)

# Add & Norm
dx2 = Add()([dx1, dx2])
dx2 = LayerNormalization(axis=-1, epsilon=1e-5)(dx2)

#FeedForward
decoder_feed_forward = Dense(units=FF_DIM, activation="relu")(dx2)
decoder_feed_forward = Dense(units=EMBEDDING_DIM)(decoder_feed_forward)

# Add & Norm
decoder_output = Add()([dx2, decoder_feed_forward])
decoder1_output = LayerNormalization(axis=-1, epsilon=1e-5)(decoder_output)

# --------------------------------------------------
# Decoder 블록 2
# --------------------------------------------------

decoder_masked_attention2 = MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=KEY_DIM,
    dropout=0.1,
    name="decoder2_masked_self_attention",
)(
    query=decoder1_output,
    key=decoder1_output,
    value=decoder1_output,
    use_causal_mask=True,
)

dx2 = Add(name="decoder2_masked_attention_add")([
    decoder1_output,
    decoder_masked_attention2,
])

dx2 = LayerNormalization(
    axis=-1,
    epsilon=1e-5,
    name="decoder2_masked_attention_norm",
)(dx2)

decoder_cross_attention2 = MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=KEY_DIM,
    dropout=0.1,
    name="decoder2_cross_attention",
)(
    query=dx2,
    key=encoder2_output,
    value=encoder2_output,
)

dx2_cross = Add(name="decoder2_cross_attention_add")([
    dx2,
    decoder_cross_attention2,
])

dx2_cross = LayerNormalization(
    axis=-1,
    epsilon=1e-5,
    name="decoder2_cross_attention_norm",
)(dx2_cross)

decoder_ffn2 = Dense(
    units=FF_DIM,
    activation="relu",
    name="decoder2_ffn_expand",
)(dx2_cross)

decoder_ffn2 = Dense(
    units=EMBEDDING_DIM,
    name="decoder2_ffn_restore",
)(decoder_ffn2)

decoder_ffn2 = Dropout(
    rate=0.1,
    name="decoder2_ffn_dropout",
)(decoder_ffn2)

decoder2_output = Add(name="decoder2_ffn_add")([
    dx2_cross,
    decoder_ffn2,
])

decoder2_output = LayerNormalization(
    axis=-1,
    epsilon=1e-5,
    name="decoder2_ffn_norm",
)(decoder2_output)

# Linear
logits = Dense(units=GERMAN_VOCAB_SIZE, name="output_linear")(decoder2_output)
# (None, 40, GERMAN_VOCAB_SIZE)

# Softmax
outputs = Softmax(axis=-1,name="output_softmax")(logits)

model = Model(inputs=[encoder_inputs,decoder_inputs], outputs=outputs)

# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])

es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath="./_save/keras_Attention_Eng_to_German01.keras", monitor="val_loss",mode="min", save_best_only=True)

start = time.time()
model.fit(x=[encoder_train_tokens, decoder_train_inputs], y=decoder_train_labels, sample_weight=train_sample_weights, batch_size=16, epochs=50, callbacks=[es, mcp], validation_data=([encoder_val_tokens, decoder_val_inputs], decoder_val_labels, val_sample_weights), shuffle=True)
end = time.time()

real_time = np.round(end-start, 4)

# 4 평가 및 예측
test_result = model.evaluate(
    x=[encoder_test_tokens, decoder_test_inputs],
    y=decoder_test_labels,
    sample_weight=test_sample_weights,
    batch_size=64,
    return_dict=True,
)

print("테스트 결과:", test_result)

def translate_sentence(english_sentence):
    """영어 문장 하나를 독일어로 순차 생성한다."""

    encoder_tokens = english_vectorizer(
        tf.constant([english_sentence])
    )

    decoded_words = ["[start]"]

    for position in range(MAXLEN):
        decoder_text = " ".join(decoded_words)

        # German vectorizer는 41개를 출력하므로 마지막 열을 제거해
        # 모델의 Decoder 입력 길이 40개로 맞춘다.
        decoder_tokens = german_vectorizer(
            tf.constant([decoder_text])
        )[:, :-1]

        predictions = model.predict(
            [encoder_tokens, decoder_tokens],
            verbose=0,
        )

        next_token_id = int(
            tf.argmax(
                predictions[0, position, :],
                axis=-1,
            )
        )

        next_word = german_vocabulary[next_token_id]

        if next_word == "[end]":
            break

        decoded_words.append(next_word)

    return " ".join(decoded_words[1:])


for sample_index in range(5):
    english_sentence = x_test.iloc[sample_index]
    real_german_sentence = y_test.iloc[sample_index]
    predicted_german_sentence = translate_sentence(
        english_sentence
    )

    print("\n영어:", english_sentence)
    print("실제 독일어:", real_german_sentence)
    print("예측 독일어:", predicted_german_sentence)