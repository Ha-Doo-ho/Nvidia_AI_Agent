import tensorflow as tf
import numpy as np
import time
from csv import QUOTE_NONE
import pandas as pd 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold, KFold
import re # 정규표현식 지원
from keras.layers import TextVectorization
from keras_hub.layers import TokenAndPositionEmbedding
import string

# 1-1 데이터
NUM_SAMPLES = 50000

PATH = "./_data/deu.txt"

# 1-2 데이터 전처리
# pandas의 read_csv 기능으로 txt 파일이 바로 데이터프레임으로 만들 수 있다. 
data_csv = pd.read_csv(filepath_or_buffer=PATH,
                   sep="\t",
                   header=None, # 파일에 열 이름이 없을 때 사용
                   names=["English", "German"],
                   dtype=str,
                   encoding="utf-8",
                   keep_default_na=False, #"NA" 같은 문장을 결측값으로 바꾸지 않음
                   quoting=QUOTE_NONE) #문장 속 따옴표를 CSV 제어문자가 아닌 일반 문자로 처리

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
data_csv["German"] = "[start]" + data_csv["German"] + "[end]"

# x: Encoder 입력. y: Decorder용 독일어 문장 
x = data_csv["English"]
y = data_csv["German"] 

# 홀드아웃 방식으로 데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=11, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=11, shuffle=True)

print(data_csv.shape)                # (50000, 2)
print(x_train.shape, y_train.shape)  # (28000,) (28000,)
print(x_val.shape, y_val.shape)      # (12000,) (12000,)
print(x_test.shape, y_train.shape)   # (10000,) (28000,)

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
Encoder가 영어 의미를 만들고 Decoder가 독일어를 순서대로 생성   
"""
