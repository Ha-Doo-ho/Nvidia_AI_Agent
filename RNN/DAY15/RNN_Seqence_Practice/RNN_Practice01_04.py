import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time 
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling1D, Embedding

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import AUC
from keras.datasets import imdb

"""
RNN에서 문자 임베딩을 하기 위해 필요한 것은 --> 패딩 혹은 truncate / 마스킹
정헤진 길이로 맞추어야 하는데, 만약 문장 길이가 정해진 길이보다 짧으면, 패딩이며(앞 혹은 뒤) 길면 truncating(앞 혹은 뒤) 으로 잘라야 함. 
패딩과 마스킹은 엄연히 다르다. 패딩은 그저 채워주기만 할 뿐 Embedding할 때 이건 읽지 말라고 하네스를 걸진 않는다. 
그래서 패딩된 것을 읽지 말라고 마스킹을 하는 것이다.  
"""
 
VOCAB_SIZE = 10000  # 사전에 등록된 단어는 몇 종류로 할 것인가? 즉 서로 다른 단어 종류를 몇 개 사용할 것인가?
MAX_LENGTH = 200    # 문장의 길이를 얼마로 정할 것인가? 문장은 단어를 ID화 한 것으로 이루어져 있다.(여기는 그렇게 되어 있음) 
EMBEDDING_DIM = 128 # 하나의 단어를 만들기 위해 몇차원의 Vector을 사용할 것인가? "안녕" 이란 단어를 만들기 위해 128차원 Vector을 사용한다는 것이다. 
PATH = "./_save/keras_RNN_imdb02.keras"

# 1 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE) #num_word:  빈도에 따라 순위가 매겨집니다. (학습 데이터셋에서) 얼마나 자주 나타나는지를 고려하여, 가장 빈번하게 나타나는 단어 TOP 10000개를 사전에 등록한다. 
#print(x_train.shape, y_train.shape) #(25000,) (25000,)
#print(x_train) #[list([1, 14, 22, 1, . . ., 78, 32]) . . . )]   --->  [ [ ] ] 2차원 형태이다. 
print(x_train[0][:20]) #[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25] 행은 문장 열은 그 문장안에 있는 단어를 나타낸다.
print(y_train[:20]) # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

# 1-2 데이터 전처리 --> 입력값인 x를 전처리 한다는 것을 늘 잊지 말 것.
# x_train을 전처리 하는데, padding할 크기는 200단어로 maxlen을 걸어놓았음. prePadding을 실행할 것이며 만약 200단어보다 크면 truncating을 할 건데, 뒤에서부터 자를 것이다.
x_train = pad_sequences(sequences = x_train, maxlen=MAX_LENGTH, padding="pre", truncating="post")
x_test = pad_sequences(sequences = x_test, maxlen=MAX_LENGTH, padding="pre", truncating="post")

# train_test 이지만 메서드 이름에 국한되어서는 안된다. train 중에 test는 validation이 될 수 있다. 이것을 명심해야 한다. 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, shuffle=True, random_state=11, stratify=y_train)

# 2 모델 구성
model = Sequential()
model.add(Input(shape=(MAX_LENGTH, ))) # 입력은 무조건 벡터를 요구한다. 입력 단어의 개수를 요구한다. 단어 200 length까지 입력될 수 있다. 그래셔 입력이 이렇다. 

model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True))
# input_dim: Integer. Size of the vocabulary
# output_dim: Integer. Dimension of the dense embedding. --> 여기서 각 단어를 표현할 때 몇 차원의 벡터로 표현하는지를 결정한다. 

model.add(GlobalAveragePooling1D()) #모든 문장을 평균낸다. 그리고 128길이의 하나의 벡터로 바꿔준다. 즉 속도를 더 빠르게 해주지만 여기서도 단어의 순서를 잃어버리게 만든다.
model.add(Dense(64, activation="relu")) 
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid")) # 여기서 이게 부정인지(0) 긍정인지(1) 0초과 1미만의 값을 갖는 sigmoid를 통해 알 수 있다. 보통 0.5를 넘으면 긍정이라고 볼 수 있을 것이다. 
model.summary()

"""

단계        코드                        출력 shape
------------------------------------------------------------
입력	    Input(shape=(200,))	        (64, 200)
임베딩	    Embedding(10000, 128)	    (64, 200, 128)
평균 풀링	GlobalAveragePooling1D()	(64, 128)

200개의 단어 위치, 문장들은 다 사라지고 200개의 문장을 통합한 어떠한 단어가 만들어지고, 그 단어를 내가 원하는 차원의 벡터로 변환한다. --> GlobalAveragePooling1D() Vector니까 1D이고 이게 속도를 ㅈㄴ 빠르게 만드는 이유이다. 

"""

# 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])
es = EarlyStopping(monitor="val_accuracy", patience=30, mode="max",restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_accuracy", mode="max", save_best_only=True)

start_time = time.time()
model.fit(x_train, y_train, batch_size=64, epochs=100, callbacks=[es, mcp], validation_data=(x_val, y_val), shuffle=True)
end_time = time.time()

# 4 평가 예측
loss, accuracy, auc = model.evaluate(x_test, y_test)

# 먼저 찍어볼 것. 왜?? 출력이 뭐가 나올지 알 수가 없다. 그래서 무조건 출력을 생짜로 무조건 찍어보는 습관을 가질 것. 

#y_predict = model.predict(x_test)
#print(f"{y_predict}\n 소요시간:{np.round(end_time-start_time,4)}") # 여기서 한번 flatten도 안걸고 출력을 해보아야 한다. 이게 어떻게 출력이 될지 알 수 없기 때문이다. 
# 여기서 한번 순수 출력이 무엇인지 알아내고 그 다음에 위에서 조금씩 바꾸던가 해서 원하는 값으로 바꿔야 한다. 
"""
[[0.03324829]
 [0.9873345 ]
 [0.9578764 ]
 ...
 [0.00524979]
 [0.05251684]
 [0.72891295]]
 이렇게 나온다. 2차원으로 나오며(이건 flatten() 으로 1차원으로 펴던가 전처리가 필요하다는 뜻) 
 뜻은 첫번째 리뷰가 긍정일 확률이 3.3%라는 듯이다. 두번째 리뷰가 긍정일 확률은 98.7%라는 뜻이다. Binary분류니까 전자는 부정, 후자는 긍정이라고 볼 수 있다. 
 이렇게 나오는 것을 미리 보아야 하기 대문에 print로 생짜로 찍어보라는 갓이다. 
"""

y_predict = model.predict(x_test).flatten()
y_predict_probablity = (y_predict > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_predict_probablity)
labels = ["Negative", "Positive"]
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(cmap="Oranges")
plt.show()

