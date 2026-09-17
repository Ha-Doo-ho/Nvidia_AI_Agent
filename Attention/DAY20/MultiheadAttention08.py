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


""" 
FFN 이 무엇인가? 
이 렐루(ReLU) 오븐이 컴퓨터 안에서 어떻게 마법을 부리는지 3단계로 아주 쉽게 알려줄게요!

트랜스포머 모델 안에서 단어(토큰)들은 먼저 어텐션이라는 시간을 가집니다. 이건 단어들이 다 같이 모여서 회의를 하는 것과 같아요.

예를 들어 문장에 "배"라는 단어가 있다고 해볼게요. 이 단어는 옆에 있는 친구들을 쓱 둘러봅니다. 

옆에 "바다"와 "항해"라는 단어가 있으면, "아하! 나는 먹는 과일 배가 아니라, 타는 배(Ship)구나!" 하고 주변 상황(문맥)을 파악해서 정보를 잔뜩 모읍니다. --> Attention

1. 재료는 모두 '숫자'예요
컴퓨터는 사과, 배, 바다 같은 단어를 글자가 아니라 '숫자'로 기억해요. 어텐션(회의)을 끝내고 온 정보들은 요리(계산)를 거치면서 플러스(+) 숫자가 되기도 하고, 마이너스(-) 숫자가 되기도 해요.

    "이 단어는 긍정적인 뜻인 것 같아!" ➔ +10 (좋은 재료)

    "이 단어는 명사가 아니라 동사 같아!" ➔ -5 (방해되는 재료)

2. 렐루(ReLU) 오븐의 단호한 마법 규칙!
이 숫자 재료들이 렐루 오븐에 들어가면, 딱 하나의 자비 없는 규칙이 작동해요.

    "0보다 작은 숫자(마이너스)는 전부 0으로 만들어 버려라! 0보다 큰 숫자(플러스)는 그대로 살려둬라!"

    +10점짜리 정보 ➔ 오븐 통과 ➔ 그대로 +10

    -5점짜리 정보 ➔ 오븐 통과 ➔ 0 (삭제!)

    +3점짜리 정보 ➔ 오븐 통과 ➔ 그대로 +3

즉, "쓸모없거나 방해되는 마이너스 정보는 쓰레기통에 버리고(0으로 만들고), 진짜 중요한 플러스 정보만 남기는 마법"이에요.

3. 이게 왜 '비선형(매직)'일까요?
    그냥 계속 더하고 곱하기만 하면(선형), 좋은 재료와 나쁜 재료가 하나의 냄비 안에서 계속 뒤섞이게 돼요. 나중에는 이게 무슨 맛인지 알 수 없는 꿀꿀이죽이 되어버리죠.

    하지만 렐루 오븐이 중간에 개입해서 "잠깐! 마이너스 맛이 나는 건 다 빼버려!" 하고 과감하게 0으로 싹둑 잘라버리면 어떻게 될까요?

    쓸데없는 정보는 사라지고, 단어가 가진 '진짜 중요한 특징'만 뾰족하게 살아남게 돼요. 이렇게 0을 기준으로 규칙이 확 바뀌는 걸 어른들의 어려운 말로 '비선형(Non-linear)'이라고 부른답니다.

💡 한 줄 요약!
컴퓨터 안의 매직 오븐(비선형 함수)은 "도움이 안 되는 마이너스 정보는 '0'으로 지워버리고, 중요한 플러스 정보만 살려서 인공지능이 똑똑한 결정을 내리게 도와주는 강력한 필터" 역할을 한답니다!
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LayerNormalization은 무엇인가?
1. 문제 발생: 목소리 크기가 다 제멋대로야!
    요리(계산)를 여러 번 반복하다 보면, 단어들이 가진 숫자(정보)들의 크기가 뒤죽박죽이 됩니다.

    어떤 숫자는 10,000처럼 엄청 커지고, 어떤 숫자는 -0.001처럼 너무 작아져요. 이건 마치 오케스트라 연주를 하는데, 큰북은 귀청이 떨어지게 쾅쾅 치고 플루트는 개미 목소리만큼 작게 부는 것과 같아요.

    이대로 다음 단계로 넘어가면, 컴퓨터는 너무 큰 숫자에만 깜짝 놀라서 작은 숫자의 중요한 정보는 무시해 버리고 맙니다. 학습이 엉망진창이 되는 거죠.

2. 해결책: '스마트 볼륨 조절기'의 등장 (Layer Normalization)
    이때 '레이어 정규화'라는 똑똑한 볼륨 조절기가 등장합니다. 이 조절기는 단어 하나가 들고 있는 수백 개의 숫자들을 쫙 훑어본 다음, 컴퓨터가 딱 듣기 편안한 소리 크기로 맞춰줍니다.

3. 볼륨을 조절하는 2가지 마법 규칙
    이 스마트 볼륨 조절기는 정확히 두 가지 행동을 합니다.

    중간으로 모으기 (평균을 0으로): 숫자들의 전체적인 덩치가 너무 크거나 너무 작으면, 딱 중간 기준점을 0으로 끌고 와서 영점 조절을 해줍니다.

    간격 예쁘게 맞추기 (분산을 1로): 숫자들이 너무 심하게 들쭉날쭉하게 퍼져있으면 간격을 좁히고, 너무 답답하게 뭉쳐있으면 살짝 벌려줍니다. 흩어진 정도를 딱 1이라는 예쁜 간격으로 맞춰요.

🎯 결론: 그래서 왜 하는 건가요?
레이어 정규화는 "아무리 복잡한 마법 요리(계산)를 수십 번 반복해도, 정보들이 폭주하거나 사라지지 않도록 꽉 잡아주는 안전장치"입니다.

이 볼륨 조절기 덕분에 큰북 소리도, 플루트 소리도 선명하게 잘 들리게 되어서, 트랜스포머 모델이 수백 층의 블록을 높게 쌓아도 무너지지 않고 똑똑하게 학습할 수 있는 거랍니다!

모든 악기의 볼륨을 중간(0)으로 맞추고, 소리의 크기 변화(1)를 똑같이 만들어버리면 튀는 소리는 없어지지만, 음악이 마치 로봇이 부르는 것처럼 엄청 밋밋하고 지루해지겠죠? 슬픈 노래든 신나는 노래든 다 똑같은 느낌이 될 테니까요.

그래서 컴퓨터 안의 오디오 감독님은 소리를 깔끔하게 정리한 직후에, 각 단어의 개성을 다시 예쁘게 살려주기 위해 '감마(Gamma)'와 '베타(Beta)'라는 두 개의 마법 다이얼을 만지기 시작합니다.

1. 감마(Gamma): "너만의 강약을 살려줄게!" (곱하기 다이얼)
    볼륨을 다 똑같이 '1'의 간격으로 맞춰놨더니, 천둥소리처럼 쾅! 하고 터져야 하는 중요한 정보마저 얌전해졌어요. 이때 감마(Gamma) 다이얼을 돌립니다.

    어떻게 작동하나요? 정규화된 숫자에 특정 값을 '곱해주는(x)' 역할을 해요.

    어떤 효과가 있나요? "이 단어는 진짜 중요한 랩 파트니까, 다시 간격을 3배로 쫙 늘려서(x3) 다이나믹하게 만들어주자!" 하고 원하는 만큼만 개성(대비)을 쫙 증폭시켜 줍니다. 반대로 얌전해야 할 파트는 0.5를 곱해서 더 부드럽게 만들 수도 있죠.

2. 베타(Beta): "너만의 분위기를 만들어줄게!" (더하기 다이얼)
    평균을 다 똑같이 '0'으로 맞춰놨더니, 전체적으로 밝고 높은음이어야 하는 정보도 무조건 중간으로 내려와 버렸어요. 이때 베타(Beta) 다이얼을 누릅니다.

    어떻게 작동하나요? 숫자에 특정 값을 '더해주는(+)' 역할을 해요.

    어떤 효과가 있나요? "이 단어는 전체적으로 긍정적이고 밝은 분위기니까, 기준점을 0에서 +5로 쑥 올려주자!" 하고 소리의 전체적인 위치(기본 톤)를 위아래로 이동시켜 줍니다.

🎯 요약: 왜 굳이 '정리'했다가 다시 '개성'을 주나요?
    처음부터 엉망진창인 상태로 놔두면 컴퓨터가 너무 헷갈려서 학습을 포기해 버려요.

    그래서 먼저 ① 레이어 정규화로 모든 도화지를 새하얗고 반듯하게(0과 1로) 청소해 줍니다. 그래야 컴퓨터가 안심하고 계산을 계속할 수 있거든요.

    그러고 나서 ② 감마와 베타라는 색연필을 꺼내서, "여기엔 빨간색(감마)을 좀 진하게 칠하고, 위치는 살짝 오른쪽(베타)으로 옮기는 게 좋겠어!"라며 깔끔한 바탕 위에 가장 완벽한 비율로 개성을 다시 그려 넣는 것이랍니다.

    가장 신기한 건 이 '감마'와 '베타' 다이얼을 얼만큼 돌릴지 사람이 정해주는 게 아니라, 인공지능이 수많은 공부를 하면서 "아, 여기선 감마를 2배로 올리는 게 정답이구나!" 하고 스스로 깨닫고 알아서 다이얼을 돌린다는 점이에요!
"""

#Attention is all you need 구현

# 1 데이터
VOCAB = 10000
MAXLEN = 200
EMBEDDING_DIM = 128
FF_DIM = 256 #피드 포워드 디멘션
PATH = "./_save/keras_Attention_imdb5.keras"
(x_train, y_train), (x_test, y_test)= imdb.load_data(num_words=VOCAB)

# 1-2 데이터 전처리 
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, dtype="int32", padding="pre", truncating="pre")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=11, shuffle=True, stratify=y_train)

x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_val = pad_sequences(sequences=x_val, dtype="int32" , maxlen=MAXLEN, padding="pre", truncating="pre")

# 2 모델 구성
inputs = Input(shape=(MAXLEN, ))

# 단어 / 포지션 임베딩 
TP_EMBEDDING = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM,mask_zero=True)(inputs)

# MultiheadAttention을 구성함
# num_heads가 3이므로  W₁ᵠ. . . Wₙᵠ | W₁ᴷ. . .Wₙᴷ | W₁ⱽ. . . Wₙⱽ 환경인데, n이 3일 것이고, 각 가중치는 Q,K,V를 64차원으로 만들 가중치 행렬을 제작할 것이다. 
# 옆에 "바다"와 "항해"라는 단어가 있으면, "아하! 나는 먹는 과일 배가 아니라, 타는 배(Ship)구나!" 하고 주변 상황(문맥)을 파악해서 정보를 잔뜩 모으는 것이다. 여기서 보통 얻을 걸 다 얻는다. 
attention_output1 = MultiHeadAttention(num_heads=3, key_dim=64)(query=TP_EMBEDDING, key=TP_EMBEDDING, value=TP_EMBEDDING)

# 잔차 연결 (Residual Connection) 어텐션에서 여러 단어들에 문맥이 고려된 정보가 이미 포함되었을 수 있다. 이때, 층이 깊어지면 원본의 내용이 변질될 위험이 있다. 
# ResNet의 핵심 아이디어인 각 계층, 대리-> 과장 -> 이사 . . . 사장 을 거치는 것이 아닌 사장으로 한번에 갈 수 있는 경로를 제공하는 것. 
# 이 직접 경로가 핵심이다.  이 덕분에 모델이 깊어질수록, 원본을 참조 하지 못하는 경우를 최대한 줄일 수 있다. 
# 기울기 소실 문제를 줄일 수 있다.(직접적인 정보·기울기 전달 경로를 제공하기 때문이다.) 깊은 신경망의 학습을 안정화하기 위해 사용한다.
x1 = Add()([TP_EMBEDDING, attention_output1]) # 혹은 그냥 x = TP_EMBEDDING + attention_output 해도 된다. 

# LayerNormalization
# LayerNormalization은 “각 토큰 벡터의 숫자들을 일정한 기준으로 조정” 한다. 더 정확하게 말한다면, 128개의 값을 평균 0, 표편 1에 가깝게 표준화한다.
# CNN의 BatchNormalization과 동일한 역할을 한다고 보면 된다. 잔차 연결과 함께, 학습을 안정화 하고 학습 가속화를 위해 사용한다. 물론 정규화는 늘 원본의 내용을 깎아낸다는 단점이 있다.
x1 = LayerNormalization(epsilon=1e-5)(x1)

# Feed Forword
# 단어간 관계 파악은 Attention에서 상당 부분 완료된다. 여기서는 단어 간 관계 파악을 넘어 각 토큰(단어)의 표현(차원)을 정교하게 비선형 변환하는 데, 사용한다.
# 그 방법이 차원을 늘려서 활성화 함수 적용하고 다시 원래대로 되돌리는 것이다.
# 역시 Attention all you need에 들어가 있다. 
ffn_output = Dense(units=FF_DIM, activation="relu")(x1)
ffn_output = Dense(units=EMBEDDING_DIM)(ffn_output)
ffn_output = Dropout(0.2)(ffn_output)

# 2번째 Residual_Connection 및 LayerNormalization
x1 = x1 + ffn_output
x1 = LayerNormalization(axis=-1, epsilon=1e-4)(x1)

# ==================================================
# Encoder 블록 2
# ==================================================
attention_output2 = MultiHeadAttention(num_heads=3, key_dim=64, name="encoder2_self_attention")(query=x1, key=x1, value=x1)

x2 = Add(name="encoder2_attention_add")([x1, attention_output2])
x2 = LayerNormalization(axis=-1, epsilon=1e-5)(x2)

ffn_output2 = Dense(units=FF_DIM, activation="relu")(x2)
ffn_output2 = Dense(units=EMBEDDING_DIM)(ffn_output2)

x2 = Add(name="Add & Norm")([x2, ffn_output2])
encoder_output = LayerNormalization(name="add & Norm", axis=-1, epsilon=1e-5)(x2)

#-------------------------------------------------------------------------------------------------- 인코더 2개 했을 때의 끝

#-------------------------------------------------------------------------------------------------- 디코더의 시작
# 디코더는 인코더가 이해한 정보를 바탕으로 최종 출력 결과(텍스트 등)를 순차적으로 생성(Generation)하는 기능을 맡는다. Attention is all you need 에 같이 들어 있다.

"""
주요 기능 및 특징 

자기회귀(Auto-regressive) 생성: 이전에 생성된 토큰들을 입력으로 삼아 다음 토큰의 조건부 확률을 하나씩 예측하고 완성해 나갑니다. 

마스크드 셀프 어텐션(Masked Self-Attention): 생성 과정에서 미래의 토큰 정보를 미리 보지 못하도록(Information Leakage 방지) 가려주는 역할을 합니다.

인코더-디코더 어텐션(Cross-Attention): 인코더가 추출한 입력 문장의 맥락 정보와 디코더가 생성 중인 현재 상태를 연결하여 연관성이 높은 입력 정보를 참고하게 만듭니다

확률 분포 출력: 선형 레이어(Linear Layer)와 소프트맥스(Softmax)를 거쳐 다음에 올 가장 적절한 토큰의 확률을 계산합니다
"""
decoder_inputs = Input(shape=(MAXLEN, ), dtype="int32")

# Decorder의 단어 임배딩 + 위치 임베딩
decoder_embedding = TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(decoder_inputs)

#마스크드 셀프 어텐션(Masked Self-Attention): 생성 과정에서 미래의 토큰 정보를 미리 보지 못하도록(Information Leakage 방지) 가려주는 역할을 합니다.
masked_attention_output = MultiHeadAttention(num_heads=3, key_dim=64,)(query=decoder_embedding, key=decoder_embedding, value=decoder_embedding, use_causal_mask=True) #use_causal_mask (인과 마스크)

decoder_x = Add(name="Add & Norm")([decoder_embedding, masked_attention_output])
decoder_x= LayerNormalization(name="Add & Norm", axis=-1, epsilon=1e-5)(decoder_x)

# Decorder의 Multi-HeadAttention 2번째 --> Cross-Attention이라고도 부른다. 인코더의 결과와 디코더의 결과를 행렬 덧샘 하기 때문이다.
cross_attention_output = MultiHeadAttention(num_heads=3, key_dim=62, name="decoder_cross_attention")(query=decoder_x, key=encoder_output, value=encoder_output)

cross_attention_output = Add(name="Add & Norm")([cross_attention_output, decoder_x])
cross_attention_output = LayerNormalization(axis=-1, epsilon=1e-5,)(cross_attention_output)

#Feed Forward
Decoder_Feed_Forward = Dense(units=FF_DIM, activation="relu")(cross_attention_output)
Decoder_Feed_Forward = Dense(units=EMBEDDING_DIM, activation="relu")(Decoder_Feed_Forward)

#Add & Norm
Decorder_output = Add(name="Add & Norm")([Decoder_Feed_Forward, cross_attention_output])
Decorder_output = LayerNormalization(axis=-1, epsilon=1e-5, name="Add & Norm")(Decorder_output)

##########################################################################---> 디코더의 끝

Decorder_output = GlobalAveragePooling1D()(Decorder_output)
Decorder_output = Dense(units=64, activation="relu")()
Decorder_output = Dropout(0.3)(Decorder_output)

outputs = Dense(units=1, activation="sigmoid")(Decorder_output)
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