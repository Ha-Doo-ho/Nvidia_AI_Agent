"""
1. 왜 Self-Attention이 나왔는가?
앞서 배운 LSTM이나 양방향(Bidirectional) LSTM은 문장을 앞에서부터 뒤로, 혹은 양쪽에서 가운데로 ★'순서대로'★ 읽었다. --> 앞이건 뒤건 순서대로 읽었기에 문제가 생겼다. 
이 방식의 한계는 문장이 길어지면 단어와 단어 사이의 물리적 거리가 멀어져 관계를 파악하기 힘들다는 것이다. 

나는 어제 친구들과 만나서 밥을 먹고 카페에 가서 수다를 떨다가 집에 왔다.
여기서 왔다의 주체가 '나' 라는 것을 알기 위해 모델은 수많은 단어를 순차적으로 거쳐야만 했고(기존에 RNN 기반 모든 모델), 그 과정에서 정보가 흐려졌다. 

또한 순차적으로 처리하므로 병렬 처리가 불가능하여 긴 문장에서는 정보 흐려짐 + 속도 저하도 같이 발생한다. 

구글은 여기서 혁명적인 질문을 하였다. 
"단어를 굳이 순서대로 읽어야 해?? 문장 속 모든 단어들을 한 번에 펼쳐놓고, 서로가 서로에게 얼마나 중요한지 그 '관계 비율'만 계산하면 안 될까?"

즉 "왔다" 와 관계가 가장 깊은 것 중 하나는 "나" 임을 알면 되는 것 아닐까??

2. Self-Attention의 3가지 핵심 요소 (Q, K, V) --> 이 3개는 모두 각각 벡터이다.
Self-Attention의 메커니즘을 구글 논문은 데이터베이스 검색 시스템에 비유하여 Q(Query), K(Key), V(Value) 라는 3가지 요소로 명확하게 정의한다. 
입력된 각 토큰(단어)의 임베딩 벡터에, 랜덤 생성된 서로 다른 학습 가중치 행렬(Wᵠ, Wᴷ, Wⱽ) 을 곱해 만들어 진다. 이들이 임베딩된 단어 벡터들과 곱해져서 Q, K, V가 만들어진다.

수십 명의 사람들이 모인 파티를 상상해 보자. 모든 단어(사람)는 파티장에 입장할 때 다음 3가지를 부여받는다.
    1. Q(Query Vector, 질문표): 내가 지금 찾고 있는 사람의 조건이다. (예: "나는 동사를 수식하는 부사나 명사를 찾고 있어! 그런 조건을 만족하는 단어는 누구야?")
    2. K(Key, 이름표): 내가 어떤 사람인지 나타내는 키워드이다. (예: "나는 장소를 나타내는 명사야.")
    3. V(Value, 진짜 정보): 내가 실제로 품고 있는 구체적인 의미와 뉘앙스이다. 
    

3. 논문의 실제 작동 방식: 단어들의 소개팅
논문에서 가장 유명한 예시 문장을 하나 보겠다.
"The animal didn't cross the street because it was too tired." (그 동물은 길을 건너지 않았다. 왜냐하면 그것은 너무 피곤했기 때문이다.)

사람은 그것(it)이 길(street)이 아니라 동물(animal)이라는 것을 본능적으로 안다. 하지만 인공지능은 이를 어떻게 알아내는가? --> 여기서 Self Attention이 작동한다.

1. 질문 던지기 (Q x K연산): '그것(it)'이라는 단어가 자신의 Q(질문표--> EMBEDING 벡터 차원 보다 훨씬 작은 차원의 Vector이다.)를 든다. 가중치 행렬을 임베딩된 행렬에 곱하면 Q(질문표) 벡터가 나온다.
   "나랑 의미가 가장 자연스럽게 이어지는 명사는 누구야?" 그리고 문장 속 모든 단어들의 K(이름표)와 대조해 본다. --> 유사도 검사니까 여기서 내적을 사용한다.
2. 관계 점수 계산 (Attention Score): 그것(it)의 Q는 길(street)의 K와 대조했을 때보다 '동물(animal)'의 K와 대조했을 때 수학적으로 훨씬 높은 점수(유사도)를 기록한다. 피곤할 수 있는 주체는 동물이기 때문이다.
   Query Space에 찍힌 벡터와 Key Space에 찍힌 벡터의 위치가 유사하면 (즉 내적 값이 크다면) 그 질문에 대한 답일 확률이 높다는 것이다.
   이 내적 값이 크면 유사한다고 말하고 다른 말로 Key(답)는 Query(질문)를 주목(ATTEND)한다고 말한다. 

아마 이렇게 했을 것이다. 
street의 K: "나는 장소를 나타내는 명사야."  
animal의 K: "나는 동물을 나타내는 명사야."

3. 정보 흡수 (Softmax와 V의 곱)
점수를 비율(0~1 사이의 확률)로 변환 (Softmax이다. 각 확률을 다 더해야 1이나온다.) 예컨대 '동물' 90%, '길' 10%의 결과가 나왔다고 가정해 보자. 그러면 '그것(it)'은 동물의 V(진짜 정보)를 90%만큼, 길의 V를 10%만큼 섞어서 흡수한다.

4. 결론
결과적으로 '그것(it)'이라는 단어는 단순한 대명사가 아니라, 문장 안에서 관계를 맺고 있는 '동물'의 의미를 잔뜩 머금은 풍부한 문맥적 벡터로 재탄생한다. 
자기 자신(Self)의 문장 안에서 주의(Attention)를 기울일 단어를 찾아냈다는 뜻

"""

import tensorflow as tf
import numpy as np
import time 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, MultiHeadAttention, GlobalAveragePooling1D, Dense, Dropout, Token
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 1-1. Data
VOCAB = 10000
MAXLEN = 200
EMBEDDING_DIM = 128 
Path = "./_save/keras_Attention_imdb1.keras"

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)

# 1-2. Data Preprocessing 
x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre") #<-- 오직 0으로 채워주거나 padding과 truncating을 명시할 뿐이다. 
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, padding="pre", truncating="pre")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=11, shuffle=True, stratify=y_train)

# 2. Model Configuration
# 여기서 기존의 Sequential 이 아닌 Model이 사용된다.
# 오직 이전 층의 결과 전체를 다음 층으로 통째로 넘기기만 하는 Sequential 모델로는 데이터를 3갈래로 분기시켜 하나의 층에 다중인자로 넣는 구조를 깔끔하게 설계할 수 없다.
# Sequential 을 만들 때 사용하는 .add()를 사용하지 못하므로 수동으로 직접 만든다. model.add(layer(내가 넣을 것)) 이 아닌, layer()(내가 넣을 것) 이렇게 만든다. 
# 오히려 더 간단해 보일 수도 있다. 
inputs = Input(shape=(200, )) #입력층: 모양을 정의하고 변수(inputs)에 할당한다.  Keras 내부: (None, 200) 이다. 즉 행은 keras가 알아서 만드는 것이고 열은 무조건 넣어주어야 한다. 그래서 행무시 열우선이다. 

# 임베딩층: 층(Layer) 뒤에 이전 층의 출력값(inputs)을 괄호로 연결해 정보의 흐름을 명시한다.
embedding = Embedding(input_dim=VOCAB, output_dim=EMBEDDING_DIM, mask_zero=True)(inputs)

# Self-Attention층 (핵심)
# num_heads: 몇 개의 관점으로 문장을 분석할 것인가? (보통 2 ~ 8게)
# key_dim: Q, K, V 벡터의 크기
# Self-Attention이므로 query, value, key 모두에 동일한 자기 자신(x)을 넣는다.
# 즉 임베딩 층을 통과한 단 하나의 데이터 흐름인 x를 복사해서 각기 다른 역할 (질문표, 이름표, 진짜 정보)로 쪼개어 동시에 넣은 것이다.
# num_heads: Q가 몇개가 있는가? "나는 동사를 수식하는 부사나 명사를 찾고 있어! 그런 조건을 만족하는 단어는 누구야? 와 같은 질문 한종류만 하지 않을 것이다. --> 컴퓨터가 알아서 해줌
# "지금 주변에 긍정적인 단어가 있나, 부정적인 단어가 있나?", "과거형을 나타내는 단어가 근처에 있나?" 와 같은 질문들도 몇개 더 해야 정확할 것이다. (문법 담당 질문, 감정 담당 질문, 시제 담당 질문)
# 그 Q가 되는 질문의 개수를 결정하는 것이 num_heads이다. 
# 가중치 행렬은 헤드마다 할당되므로 1번헤드 Wᵠ, Wᴷ, Wⱽ, 2번헤드 Wᵠ, Wᴷ, Wⱽ 이렇게 이루어 진다. 
# 이들을 Embedding된 단어에 곱해서 같은 단어지만 서로 다른 관점으로 재표현하는 것이다. 
# key_dim 은 기존 n 차원이던, 단어 임베딩을 더 작은 차원의 Query와 Key로 변환시켜 주는 것이다. 그 이유는 보통 임베딩한 차원의 절반으로 Query와 Value를 만드는 경향이 있기 때문이다. (꼭 작아지지 않아도 되는데, 보통 개발자들이 반으로 쪼갬.)
# key_dim 은 각 담당 질문의 개수라고 생각하면 된다. num_head 각각이 가지고 있는 질문이 64개라는 의미
attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(query=embedding, value=embedding, key=embedding)

embedding = GlobalAveragePooling1D()(attention_output)
embedding = Dense(64, activation="relu")(embedding)
embedding= Dropout(0.2)(embedding)
outputs = Dense(1, activation="sigmoid")(embedding)

# 모델 묶기 (시작점과 끝점을 지정)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# 컴파일 및 훈련
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=Path, monitor="val_loss", save_best_only=True, mode="min",)

start_time = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=20, callbacks=[es, mcp], validation_data=(x_val, y_val))
end_time = time.time()

real_time = np.round(end_time-start_time, 4)

# 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test).flatten()

y_predict_class = (y_predict >= 0.5).astype("int32")
print(f"loss: {loss}, accuracy: {accuracy}, auc: {auc}")
print(f"소요시간: {real_time}초")

print(y_predict_class)

labels = ["Negative","Positive"]
cm = confusion_matrix(y_test, y_predict_class)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap="Oranges", xticks_rotation=45)
plt.show()