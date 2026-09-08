# 기존에는 각 단어 간 관계는 계산하지만 단어의 위치와 순서를 알지 못하였다. 위치 임베딩은 Self-Attention과 Transformer을 연결하는 첫 번째 다리이다.
"""
복습
a fluffy blue creature roamed the verdant forest 라는 문장이 있다고 가정하자.
이것을 생성하려면 fluffy blue creature 부터 무엇인지 알아야 한다. 그러려면 fluffy blue가 무엇인지 알아본다고 가정해보자. 
creature 을 수식해주는 형용사는 무엇인가? 라고 하였을 때, 그 질문을 만드는 것이 Query이다. Query는 임베딩된 벡터보다 더 작은 차원을 가진 벡터이다. 
Query는 Wᵠ에 creature 의 임베딩된 행렬을 곱해서 만든다. Key, Value도 마찬가지이다. Wᴷ에 임베딩된 행렬,  Wⱽ에 임베딩된 행렬을 각각 곱해서 만든다.
Query와 Key를 내적하여, 유사도를 구한다. 유사도가 크면 클 수록 그 단어를 더 잘 설명하는(연관성이 깊은) 단어라는 것이다. 
Wⱽ행렬과 임베딩 행렬을 곱해서 만들어진 Value행렬은 임베딩된 creature에 더한다. 그러면 다차원 공간 상에서 fluffy creature에 가까워진다. 
GPT는 Wᵠ, Wᴷ, Wⱽ 를 한 묶음으로 본다면 저게 여러개가 있을 수 있는데, 그게 헤드들이 여러개 있다고 말하는 것이다. 헤드들이 여러개 있다는 것은 곧 다양한 관점으로 그 단어를 이해하기 위한 문맥을 파악한다는 것이다.

this라는 단어의 의미
movie라는 단어의 의미
not이라는 단어의 의미
good이라는 단어의 의미

이렇게만 있다. 

this는 1번째 단어
movie는 2번째 단어
not은 good보다 앞에 있음
good은 마지막 단어

이러한 정보는 없다. --> 이 정보를 Position임베딩이 제공한다. 

실제 Attention 의 입력 벡터는 다음과 같다. --> 입력 벡터 = 단어 임베딩 + 위치 임베딩 이다.
위치 임베딩을 한다고 해서 shape는 변하지 않으니 안심해도 된다. 

단어의 의미 임베딩: (batch, 200, 128)
단어의 위치 임베딩: (       200, 128)
더한 결과:   (batch, 200, 128) --> 행렬 덧셈이므로 차원은 변경되지 않는다. 이 결과가 Self-Attention의 Q,K,V 입력이 된다.

즉 단어 임베딩은 위치 + 의미 가 합쳐진 것이다. 
"""
import numpy as np
from keras import ops #실제 NumPy와 동일한 함수명과 인자(Arguments) 구성이 똑같이 되어 있다. 
from keras.utils import pad_sequences
from keras.layers import Input, Layer, Embedding, Add, MultiHeadAttention, GlobalAveragePooling1D, Dense, Dropout
from keras_hub.layers import PositionEmbedding

from keras.models import Model

from keras.datasets import imdb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit

# 1-1데이터
VOCAB= 10000
MAX_LEN = 200
EMBEDDING_DIM = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)
#print(x_train.shape, y_train.shape) #(25000,) (25000,)

# 1-2 데이터 전처리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=11, shuffle=True, stratify=y_train)
#print(x_train.shape, x_val.shape) # (21250,) (3750,)
#print(y_train.shape, y_val.shape) # (21250,) (3750,)

x_train = pad_sequences(sequences=x_train, maxlen=MAX_LEN, padding="pre", truncating="pre",) #<-- 오직 0으로 채워주거나 padding과 truncating을 명시할 뿐이다. 
x_test = pad_sequences(sequences=x_test, maxlen=MAX_LEN, padding="pre", truncating="pre")
#print(x_train.shape, x_test.shape) #(21250, 200) (25000, 200) -> train 에 있는 문장의 개수는 21250개.  하나의 문장을 이루는 토큰의 길이는 200 --> 이거 padding 되어서 200이거나 truncating 되어서 200임.

#print(len(x_train[0])) #200 이다. 출력해 보면 실제로는 숫자 154개인데, 부족하니까 pre padding함. 즉, 앞에서 padding을 해서 200으로 maxlen을 만든 것이다. 

# 2 모델 구성
# 여기서 기존의 Sequential 이 아닌 Model이 사용된다.
# 오직 이전 층의 결과 전체를 다음 층으로 통째로 넘기기만 하는 Sequential 모델로는 데이터를 3갈래로 분기시켜 하나의 층에 다중인자로 넣는 구조를 깔끔하게 설계할 수 없다.
# Sequential 을 만들 때 사용하는 .add()를 사용하지 못하므로 수동으로 직접 만든다. model.add(layer(내가 넣을 것)) 이 아닌, layer()(내가 넣을 것) 이렇게 만든다. 
# 오히려 더 간단해 보일 수도 있다.
inputs = Input(shape=(MAX_LEN, ))
token_vectors = Embedding(input_dim=VOCAB, output_dim=EMBEDDING_DIM, mask_zero=True, name="token_embedding")(inputs) #기존에 하던 단어를 토큰으로 만든 것을 고차원 공간 상에서 의미를 가지는 벡터로 표현함

position_vectors = PositionEmbedding(sequence_length=MAX_LEN, name="position_embedding")(token_vectors) # 여기에 위치 임베딩이 추가됨. 어떠한 단어가 어디에 위치한다는 정보를 담게 된다.

embedding = Add()([token_vectors, position_vectors]) #2개 이상은 list --> 무조건 기억하기.  --> 결국  ★임베딩은 단어의 의미 벡터 + 단어의 위치 벡터를 행렬 덧셈을 한 값이다. 그래서 차원이 변하지 않음.★ 
#embedding = token_vectors + position_vectors 이렇게 해도 된다.  과정이 눈에 잘 보이니까 Add를 사용한 것이다.


# 입력 ID가 0이 아니면 실제 단어이다. 왜냐? 0으로 zero-padding되어 있기 때문이다. 그래서 여기서 검사한다. input을 0과 비교하는 것이다. 다르면 단어인데, 0이면 정말로 padding 된 것이기 때문이다. 
# Padding Mask를 사용하는 이유
# → pad_sequences가 넣은 단어 ID 0을
#   Attention과 Pooling 계산에서 제외. 즉, Padding Mask를 적용한 Self-Attention을 실행하여야 한다. Padding 과 Masking은 이전에도 강조하지만 별개이다. 마스킹이 여기서 된다. 
padding_mask = ops.not_equal(inputs, 0) 

# Self-Attention에서는 Query 하나가 각 Key를 바라볼 수 있는지 표현해야 한다.
"""
현재 Pre-Padding이라서 MAX_LEN 보다 작을 경우 앞 부분이 0으로 채워지게 된다. 

             Key 위치
             0  1  2  3  4
Query 0 PAD  0  0  0  0  0
Query 1 PAD  0  0  0  0  0
Query 2 단어 0  0  1  1  1
Query 3 단어 0  0  1  1  1
Query 4 단어 0  0  1  1  1

0 = Attention 계산 금지
1 = Attention 계산 허용
"""
query_mask = ops.expand_dims(padding_mask, axis=2) # Query는 세로로 존재한다. 각 행의 값이 열 방향으로 복제된다.
"""
원래:
False
False
True
True
True

False False False False False
False False False False False
True  True  True  True  True
True  True  True  True  True
True  True  True  True  True
"""

key_mask = ops.expand_dims(padding_mask, axis=1)   # Key는 각 열의 값이 행 방향으로 복제된다.
"""
원래:
False False True True True

False False True True True
False False True True True
False False True True True
False False True True True
False False True True True
"""

attention_mask = ops.logical_and(query_mask, key_mask)
"""
False False False False False
False False False False False
False False True  True  True
False False True  True  True
False False True  True  True
"""

# 실제 Attention 계산에 적용
attention_output = MultiHeadAttention(num_heads=3, key_dim=64)(query=embedding, key=embedding, value=embedding, attention_mask=attention_mask)