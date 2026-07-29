import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Dropout, BatchNormalization # 임포트 추가
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist

# ==========================================
# 1. 데이터 (기존과 완벽히 동일)
# ==========================================
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


# ==========================================
# 2. 모델 구성 (배치 정규화 + 드롭아웃의 완벽한 조화)
# ==========================================
cnn_model = Sequential(name="cnn_with_bn_and_dropout")
cnn_model.add(Input(shape=(28, 28, 1), name="Input_layer"))

# --- 1차 특징 추출기 ---
cnn_model.add(Conv2D(32, kernel_size=3, padding="same", activation='relu', name="conv_1"))
# 💡 [여기!] 1차 돋보기가 찾은 특징들의 숫자가 너무 튀지 않게 정돈합니다.
cnn_model.add(BatchNormalization(name="bn_1"))
cnn_model.add(MaxPool2D(pool_size=2, name="pool_1")) 

# --- 2차 특징 추출기 ---
cnn_model.add(Conv2D(64, kernel_size=3, padding="same", activation='relu', name="conv_2"))  
# 💡 [여기!] 2차 특징들도 다음 층으로 가기 전에 예쁘게 정돈합니다.
cnn_model.add(BatchNormalization(name="bn_2")) #괄호 안의 옵션을 건드리거나 수동으로 조작할 필요 없음. keras가 자동으로 처리한다. 
cnn_model.add(MaxPool2D(pool_size=2, name="pool_2"))

# --- 통역사 ---
cnn_model.add(Flatten(name="flatten"))
cnn_model.add(Dropout(0.25, name="dropout_1")) 

# --- 뇌세포 (은닉층) ---
cnn_model.add(Dense(64, activation='relu', name="hidden")) 
# 💡 [여기!] 가장 복잡한 계산을 하는 뇌세포의 결괏값도 정돈해 줍니다.
cnn_model.add(BatchNormalization(name="bn_3"))
cnn_model.add(Dropout(0.5, name="dropout_2")) 

# --- 출력층 ---
cnn_model.add(Dense(10, activation="softmax", name="output")) 

# ==========================================
# 3. 컴파일 및 훈련 (기존과 동일)
# ==========================================
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)

print("\n--- 훈련 시작 ---")
cnn_model.fit(
    x_train, y_train, 
    epochs=10, 
    batch_size=128, 
    validation_split=0.2, 
    callbacks=[es],
)


# ==========================================
# 4. 평가 및 예측 (기존과 동일)
# ==========================================
loss, accuracy = cnn_model.evaluate(x_test, y_test)
print(f"최종 Test Loss: {loss:.4f}")
print(f"최종 Test Accuracy: {accuracy:.4f}")

cnn_model.summary()

# 배치 정규화(Batch Normalization)이란 무엇인가?
"""
앞어 #1 데이터 전처리 단계에서 이미지 픽셀값을 255로 나누어 0~1 사이로 깔끔하게 정리했던 이유는 다음과 같다.
컨볼루션연산은 합성곱 연산이므로 수가 기하급수적으로 커질 수 있다. 학습에 문제가 생기는 것을 막는다. 
0~1의 사이값으로 고정되므로 0~255의 간격이 크게 줄어들어 노이즈를 막을 수 있다. 

이때 문제가 있다.
입력 데이터를 아무리 정규화해서 넣어주어도, 1층(Conv2D), 2층, 3층을 통과하며 가중치와 곱해지다 보면
모델 내부에서 데이터의 크기가 확 튀는 현상이 발생한다. 가중치가 곱해지고 편향이 더해지면 그렇게 될 수 있다. 
분명히 0~1 사이로 깎아서 넣었는데, 층을 겨우 몇개만 통과해도 숫자가 100 단위를 넘을 수 있다. 

<1>. 첫 번째 문제: 걷잡을 수 없는 눈덩이 효과 (Scale Shift)
우리가 전처리 단계에서 0~255인 픽셀값을 255로 나누어 0~1 사이의 얌전한 숫자로 만들어서 모델에 집어넣었다고 가정해 봅시다.
-->입력 데이터: [0.2, 0.5, 0.8] (얌전함)

1층 (Conv2D): 이 숫자들이 1층의 가중치(예: 3)와 곱해지고 편향(예: 2)이 더해집니다.
-->결과: [2.6, 3.5, 4.4] (숫자가 조금 커졌습니다)

2층 (Conv2D): 이 숫자들이 다시 2층의 가중치(예: 4)와 곱해지고 편향(예: -1)이 더해집니다.
-->결과: [9.4, 13.0, 16.6] (숫자가 확 튀기 시작합니다)

3층 (Dense): 다시 가중치(예: 10)가 곱해집니다.
--> 결과: [94.0, 130.0, 166.0]


<2>. 두 번째 문제: '내부 공변량 변화'의 진짜 의미 (움직이는 과녁)
'내부 공변량 변화(Internal Covariate Shift)'의 진짜 무서움은 단순히 숫자가 커지는 게 아니라, 학습할 때마다 숫자의 '기준점(분포)'이 이리저리 널뛰는 현상을 말합니다.

3층(Dense) 입장에서 생각해 보겠습니다. 3층은 오직 앞선 2층이 넘겨주는 데이터만 보고 정답을 맞혀야 합니다.

1 에폭(Epoch) 째:
2층이 3층에게 [10, 15, 20] 정도의 숫자를 넘겨주었습니다. 3층은 "아, 숫자가 대충 10~20 사이로 들어오는구나!" 하고 이 기준에 맞춰서 열심히 자기 가중치를 업데이트(학습)합니다.

2 에폭(Epoch) 째:
그런데 문제가 생겼습니다. 3층이 학습하는 동안, 1층과 2층도 오답 노트를 쓰면서 자기들의 가중치를 수정해 버렸습니다. 그 결과, 이번에 2층이 3층에게 넘겨주는 숫자가 갑자기 [-5, 0, 5]로 바뀌어 버렸습니다.

3층의 멘붕:
3층은 어이가 없습니다. "나는 10~20이 들어올 줄 알고 거기에 맞춰서 기껏 셋팅을 다 해놨는데, 갑자기 음수(-5)를 주면 어떡해! 처음부터 다시 학습해야 하잖아!"

즉, 각 층(Layer)은 앞 층이 넘겨주는 데이터에 적응하려고 애쓰는데, 앞 층도 계속 훈련하면서 변하기 때문에 뒤쪽 층 입장에서는 '과녁이 계속 움직이는 것'과 같은 혼란을 겪게 됩니다. 이것이 바로 '내부 공변량 변화'입니다.



💡 해결사: Batch Normalization (배치 정규화)
이 혼란을 잠재우기 위해 층과 층 사이에 BatchNormalization이라는 '군기 반장'을 세워두는 것입니다.

1 에폭 째: 2층이 [10, 15, 20]을 만들어 넘기려고 하면, 군기 반장이 개입합니다. "숫자가 너무 튀네. 평균 0, 표준편차 1로 맞춰!"
-->결과: [-1, 0, 1]로 예쁘게 다듬어져서 3층에 전달됩니다.

2 에폭 째: 2층이 가중치를 바꿔서 [-5, 0, 5]를 만들어 넘기려고 해도, 군기 반장이 또 개입합니다. "너도 평균 0, 표준편차 1로 맞춰!"
-->결과: 또다시 [-1, 0, 1]이라는 얌전한 형태로 3층에 전달됩니다.

결과적으로 3층은 앞 층이 무슨 짓을 하든 "아, 나한테 올 때는 ★★항상 평균이 0 근처인 예쁜 숫자★★로 들어오는구나"라고 안심하고 자기 학습에만 100% 집중할 수 있게 됩니다. 
이 덕분에 학습 속도가 비약적으로 빨라지는 것입니다.
"""
