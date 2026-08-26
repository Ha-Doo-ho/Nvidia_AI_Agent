import tensorflow as tf 
import pandas as pd 
import numpy as np
import time
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dense, Dropout, Flatten, LSTM, GRU, GlobalAveragePooling1D, Embedding
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
As a convention, "0" does not stand for a specific word, but instead is used to encode the pad token.
0은 패딩으로 사용한다는 것이다. RNN에서는 문장 길이를 맞추어야 하기 때문이다. 늘 강조하듯이 패딩과 마스킹은 따로 해야 한다. 
패딩된 것은 마스킹을 시켜서 의미를 부여하지 않도록 막아야 한다. 
"""

# 1 데이터
VOCAB_SIZE = 10000 # 사전에 등록된 단어는 몇 종류로 할 것인가? 즉 서로 다른 단어 종류를 몇 개 사용할 것인가?
MAX_LENGTH = 200   # 문장의 길이는 얼마나 되는지. 즉 단어를 ID로 표현하는데, ID 중복 상관 없이 length=200까지 허용하겠다는 뜻
EMBEDDING_DIM = 128   # 문장은 단어로 쪼개지고 그 단어는 컴퓨터가 못알아 먹으니까 ID로 되어 있다. 즉 단어=ID인데, 그 단어를 표현하기 위해서 몇 length의 vector로 표현할 것인가이다. 128이면 128 length의 vector로 단어 하나를 표현한다.



(x_train, y_train), (x_test, y_test) = imdb.load_data()
# print(x_train.shape, y_train.shape) #(25000,) (25000,)
# print(x_test.shape, y_test.shape)   #(25000,) (25000,)
# print(x_train) #1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, . . .단어 하나가 그에 맞는 id로 변환되어 있다. 이 id는 건들면 안된다.
print(len(x_train[0])) #218 --> 0번 리스트에 218개의 id로 문장을 만들고 있다. 다른 list는 더 많은 수의 id 로 구성되어질 수 있다는 것이다.
print(len(x_train[1])) #189 --> 1번 리스트에는 189개의 id로 문장을 만들고 있다. 
print(x_train[0][:20]) # 처음 0번부터 19번까지 다음과 같은 id로 구별하고 있다. [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25]

print(y_train[:20]) #[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]  0:부정 1:긍정
x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="pre", truncating="post") #padding: 길이가 짧으면 0을 채워넣을 건데, 그 0을 뒤에서 채울 것이냐 앞에서 채울 것이냐다. pre/post로 구별하는데, pre가 널리 쓰인다. (더 좋다고 함.)
x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="pre", truncating="post")   #truncating: 길이가 길면 자를건데, 앞에서? 아님 뒤에서? 자를 것이냐다. 이것 역시 pre/post로 구별한다. 

print("x_train.shape: ", x_train.shape) #(25000, 200) 25000개의 문장이 있고 그 문장은 200개의 id로 구별을 시키도록 통일했다. 
print("x_test.shape: ", x_test.shape)   #(25000, 200) 이건 테스트도 25000개의 문장을 가지고 하는데, 내 생각에는 100개 정도 만 test로 두고 하는게 더 좋을 것 같다. 굳이 25000개나 test로 돌릴 이유가 있을까??

# 1-2데이터 전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=11, shuffle=True, stratify=y_train)

# 2.모델 구성
model = Sequential()
model.add(Input(shape=(200, ))) # 입력 자체가 단어 ID(지금은 그 ID 종류가 200개임)가 1렬로 놓인 벡터 형태이다. 
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True, )) #안 읽는 부분은 0으로 마스킹해서 읽으면 안된다. 

model.add(GlobalAveragePooling1D())

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid")) #2진분류이다. 이게 긍정인가 부정인가를 따지니까 2진분류이며 2진분류는 Sigmoid (이건 암기해야 함. 다중은 Softmax이다.)
                                          #문장가지고 출력은 긍/부 둘 중 하나니까 마지막 출력 노드의 개수는 1이다. 
model.summary()

# 3.컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])
EarlyStopping(monitor="val_loss", )