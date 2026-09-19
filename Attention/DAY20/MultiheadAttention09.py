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

# 1-1 데이터
NUM_SAMPLES = 50000
MAXLEN = 40
VOCAB = 10000

PATH = "./_data/deu.txt"

# 1-2 데이터 전처리
# pandas의 read_csv 기능으로 txt 파일이 바로 데이터프레임으로 만들 수 있다. 
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

# 문자열 앞뒤 공백 제거 --> 이거는 파이썬 아니더라도 많이 하는 전처리니까 알아둘 것.
# "   I am tired.   " --> "I am tired." 기존 내부의 띄어쓰기는 제거 안하므로 안심할 것.
# 컴퓨터는 위의 2개를 다른 것으로 판단하여 이 전처리는 많이 한다. 데이터에는 정말 이상한 것들이 정말 많음. 
data_csv["English"] = data_csv["English"].str.strip() 
data_csv["German"] = data_csv["German"].str.strip()
# Pandas는 1차원(벡터)을 Series라고 부른다. 2차원은 DataFrame이라고 부른다. 이건 앞에서도 언급함. 
# Series가 현재, 문자열로 이루어져 있어서 str을 붙인 것이다.  
# .dt(날자/시간 접근자) .cat(범주형 접근자) 까지 3개가 보통 사용된다. 

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
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[","")
strip_chars = strip_chars.replace("]","")
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
german_train_tokens = german_vectorizer(y_train.to_numpy())
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


# 독일어 토큰을 한 칸 차이로 나누기 
# 디코더 인풋: [start] . . . 마지막 단어 
# 디코더 답지: 첫단어 . . . [end]
decoder_train_inputs = german_train_tokens[:, :-1] # 모든 행을 가져오고 열은 마지막 빼고 가져옴
decoder_train_labels = german_train_tokens[:, 1:] # 모든 행을 가져오고 0열 빼고 다 가져옴 


SAMPLE_INDEX = 0

sample_decoder_input_ids = (
    decoder_train_inputs[SAMPLE_INDEX]
    .numpy()
)

sample_decoder_label_ids = (
    decoder_train_labels[SAMPLE_INDEX]
    .numpy()
)

# 정답이 Padding 0이 아닌 위치만 확인
valid_positions = sample_decoder_label_ids != 0

valid_input_ids = sample_decoder_input_ids[
    valid_positions
]

valid_label_ids = sample_decoder_label_ids[
    valid_positions
]

decoder_input_words = [
    german_vocabulary[token_id]
    for token_id in valid_input_ids
]

decoder_label_words = [
    german_vocabulary[token_id]
    for token_id in valid_label_ids
]

print("\n===== shifted right 실제 확인 =====")
print("독일어 원문:", y_train.iloc[SAMPLE_INDEX])

print(
    "Decoder 입력:",
    decoder_input_words
)

print(
    "Decoder 정답:",
    decoder_label_words
)

# 문장 한 쌍이 Transformer를 통과하는 흐름
"""

1. 원본 문장
Encoder쪽            Decoder쪽
영어: I am tired.    독일어: Ich bin müde. 
--> 사람이 읽을 수 있는 번역 문장 쌍을 준비

2. 데이터 정리
영어: I am tired.   독일어: [start] Ich bin müde. [end]
--> 공백 · 중복(저 문장이 또 나오면 용량 잡아먹음.)을 정리하고 시작(start)과 끝(end)을 표

3. 숫자로 변환
Encoder 입력                    독일어 전체 토큰
[21, 14, 308, 0, 0, ...]        [2, 17, 34, 522, 3, 0, ...]
--> 단어를 Embedding이 처리할 수 있는 토큰 ID로 변환

4. 한칸 이동
Decoder 입력                    Decoder 정답
[start]  Ich   bin   müde       Ich    bin   müde  [end]
   2     17     34   522        17     34    522     3
--> 왼쪽 토큰들을 보고 바로 다음 토큰을 맞히도록 한 칸 이동   

5. Transformer 학습
입력 흐름
영어 → Encoder 1 → Encoder 2 → Cross-Attention

출력 흐름
앞선 독일어 → Decoder 1 → Decoder 2 → 다음 단어 확률
--> Encoder가 영어 의미를 만들고 Decoder가 독일어를 순서대로 생성   

--------------------------------------------------------------------------------------------

반드시 이해해야 하는 것
1. 영어–독일어 문장 쌍을 준비한다.

2. 문장을 정리하고, 독일어에 [start], [end]를 붙인다.

3. 문장을 토큰 번호로 바꾼다.

4. 독일어 토큰을 한 칸 어긋나게 나눈다.

5. Transformer가 다음 독일어 토큰을 예측한다.

--------------------------------------------------------------------------------------------
★★★★ 중요한 예시 ★★★★
ex)
영어: I am tired.    독일어: Ich bin müde.

Encoder 입력: I am tired.     Decoder 입력: [start] Ich bin müde.    정답: Ich bin müde. [end]

--------------------------------------------------------------------------------------------

모델이 학습하는 것은 결국 다음과 같다.
[start]를 봤으면 → Ich를 예측
[start] Ich를 봤으면 → bin을 예측
[start] Ich bin을 봤으면 → müde를 예측
[start] Ich bin müde를 봤으면 → [end]를 예측

--------------------------------------------------------------------------------------------

반드시 설명할 수 있어야 하는 것
Encoder에는 무엇이 들어가는가?
    영어 문장

Decoder에는 무엇이 들어가는가?
    정답 독일어 문장의 앞부분

Decoder의 정답은 무엇인가?
    한 칸 뒤의 독일어 토큰

Causal Mask는 왜 필요한가?
    아직 생성하지 않은 미래 단어를 보지 못하게 하기 위해서

Cross-Attention은 무엇을 연결하는가?
    Decoder의 현재 상태와 Encoder가 이해한 영어 문장

마지막 Softmax는 무엇을 출력하는가?
    다음에 나올 독일어 토큰별 확률

--------------------------------------------------------------------------------------------

Cross-Attention 부분 자세히

1. Encoder가 영어 문장 전체를 먼저 읽는다. 
영어 입력이 다음과 같다고 가정: I love this movie

Encoder는 문장 전체를 한꺼번에 처리해서 각 영어 위치의 문맥 벡터를 제작한다. 
I      → encoder_output[0]
love   → encoder_output[1]
this   → encoder_output[2]
movie  → encoder_output[3]
이때 Encoder는 번역을 시작하지 않고, 영어 문장을 분석해서 Decoder가 참고할 정보를 준비할 뿐이다. 

2. Decoder에는 [start]를 직접 넣는다.
첫 번째 Decoder 입력은 Ich가 아닌, 특별 토큰 [start]이다.
Decoder 입력: [start] 

첫 번째 Cross-Attention에서는 다음과 같이 연결된다.
Query: [start]를 처리한 Decoder 상태
Key, Value: I / love / this / movie의 Encoder 출력 전체
Decoder의 [start] 상태가 Encoder 전체를 확인한다. 이때 I 위치의 정보가 중요하다고 판단될 수 있다.

[start] 상태
      ↓ Cross-Attention
I / love / this / movie 확인
      ↓
Ich의 확률이 가장 높아진다. 그 결과 첫번째 독일어 단어인 Ich가 새엇ㅇ된다.

3. 생성된 Ich를 다음 입력에 추가한다.
이제 Decoder의 입력은 다음과 같다.
--> [start] Ich 

Decoder의 현재 상태는 단순히 Ich 하나가 아닌, 다음 문맥을 포함한다.
“번역이 시작되었고, 주어 Ich가 나왔다.”

이 상태로 다시 Encoder의 영어 문장 "전체"를 확인한다.
Decoder Query: [start] Ich
Encoder Key, Value: I / love / this / movie

이번에는 love와 관련된 정보(위치정보, 맥락정보 등. . .)를 크게 가져와 다음 단어를 예측한다.
예측 결과: liebe

이 과정의 반복이다. 
| 단계 | 지금까지 Decoder가 받은 입력     | 다음 예측 |
| ---- | ------------------------------- | -------- |
| 1    | `[start]`                       | `Ich`    |
| 2    | `[start] Ich`                   | `liebe`  |
| 3    | `[start] Ich liebe`             | `diesen` |
| 4    | `[start] Ich liebe diesen`      | `Film`   |
| 5    | `[start] Ich liebe diesen Film` | `[end]`  |

핵심
모델에게 다음 정보를 직접 알려주지는 않는다. 대신 다음만 알려준다.
영어 문장: I love this movie
독일어 정답: Ich liebe diesen Film

모델이 예측한 결과와 정답을 비교하고 역전파를 반복하면서, love가 포함된 영어 문맥과 
liebe가 나와야 하는 독일어 문맥 사이의 관계를 가중치에 학습하는 것

따라서 Cross-Attention은 처음부터 love를 찾아가는 것이 아니다. 처음에는 무작위로 보다가, 
liebe를 맞히도록 수많은 문장 쌍에서 학습되면서 love, I 등 필요한 위치에 주의를 주게 된다.


english_tokens → Encoder
decoder_inputs → Decoder
decoder_labels → Decoder가 맞혀야 할 정답
"""

# 2 모델 구성
EMBEDDING_DIM = 128
NUM_HEADS = 4
KEY_DIM = EMBEDDING_DIM // NUM_HEADS #32임
FF_DIM = 512

# 임베딩(문자 + 위치) 뒤, Dropout
encoder_inputs = Input(shape=(MAXLEN, ), dtype="int32", name="encoder_inputs")
encoder_embedding = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True, name="encoder_embedding")(encoder_inputs)
encoder_x = Dropout(rate=0.1)(encoder_embedding)

# Encoder Self_attention --> Query, Key, Value가 모두 자기 자신
attention_output1 = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(query=encoder_x, key=encoder_x, value=encoder_x)

# Add & Norm (여기서는 LayerNorm을 말함)
x1 = Add()([encoder_x, attention_output1])
x1 = LayerNormalization(axis=-1, epsilon=1e-5, name="encoder1_attention_norm")(x1)

# FeedForward
ffn_forward = Dense(units=FF_DIM, activation="relu")(x1)
ff_forward = Dense(units=EMBEDDING_DIM)(ffn_forward)
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
decoder_embedding = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=GERMAN_MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(decoder_input)

# Masked Multi-Head Attention
dx1 = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(query=decoder_embedding, key=decoder_embedding, value=decoder_embedding)

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
decoder_output = LayerNormalization(axis=-1, epsilon=1e-5)(decoder_output)

# Linear
logits = Dense(units=GERMAN_VOCAB_SIZE, name="output_linear")(decoder_output)
# (None, 40, GERMAN_VOCAB_SIZE)

outputs = Softmax(axis=-1,name="output_softmax")(logits)

model = Model(inputs=[encoder_inputs,decoder_inputs],outputs=outputs)