import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Dropout, BatchNormalization, RandomRotation
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist
from keras.models import Model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
# 🌟 L2 규제를 사용하기 위해 regularizers를 임포트합니다.
from keras import regularizers 

RandomRotation(0.08, seed=42, )
# ==========================================
# 1. 데이터 (생략 - 이전과 동일)
# ==========================================
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


# ==========================================
# 2. 모델 구성 (L2 규제 적용)
# ==========================================
cnn_model = Sequential(name="cnn_with_l2")
cnn_model.add(Input(shape=(28, 28, 1), name="Input_layer"))

# --- 1차 특징 추출기 ---
cnn_model.add(Conv2D(32, kernel_size=3, padding="same", activation='relu',))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2)) 

# --- 2차 특징 추출기 ---
# 💡 [여기!] L2 규제를 Conv2D에 적용해 봅니다. 
# 1e-4는 0.0001을 뜻하며, 규제 강도(lambda)를 나타냅니다. 작게 시작하는 것이 좋습니다.
cnn_model.add(Conv2D(64, kernel_size=3, padding="same", activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size=2))

# --- 통역사 ---
cnn_model.add(Flatten()) #GlobalAveragePool 을 넣어도 된다. 즉 cnn_model.add(GlobalAveragePool())
cnn_model.add(Dropout(0.25)) 

# --- 뇌세포 (은닉층) ---
# 💡 [여기!] 파라미터가 가장 많은 Dense 층에 L2 규제를 적용하면 효과가 좋습니다.
cnn_model.add(Dense(64, activation='relu', 
                    kernel_regularizer=regularizers.l2(1e-4))) 
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5)) 

# --- 출력층 ---
cnn_model.add(Dense(10, activation="softmax")) 


# ==========================================
# 3. 컴파일 및 훈련 
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

loss, accuracy = cnn_model.evaluate(x_test, y_test)
print(f"\n최종 Test Loss: {loss:.4f}")
print(f"최종 Test Accuracy: {accuracy:.4f}")


#4. 평가 및 예측 (이전과 동일하게 진행)
#### 혼동행렬로 평가 및 예측###
y_predict = cnn_model.predict(x_test)
y_predict_class = np.argmax(y_predict, axis=1)

cm = confusion_matrix(y_test, y_predict_class)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label (모델의 예측)', fontsize=12)
plt.ylabel('True Label (실제 정답)', fontsize=12)
plt.title('Fashion-MNIST Confusion Matrix', fontsize=15)
#plt.xticks(rotation=45) # 가로축 글씨가 겹치지 않게 45도 기울여줍니다. ticts는 눈금을 뜻한다. 
plt.tight_layout()      # 그래프 여백을 깔끔하게 자동 정리해 줍니다.
plt.show()

# 5. Grad-CAM 용 모델 적의 (추가된 부분)
print("\n--- Grad-CAM용 모델 생성 ---")

# 1. 계산 시작점: 원본 모델의 입력
inputs = cnn_model.inputs

# 2. 중간 도착점: 마지막 합성곱 층의 출력 (특징 맵)
# 위에서 지정한 이름("last_conv")으로 해당 층을 불러옵니다.
last_conv_layer = cnn_model.get_layer("conv2d_1")
last_conv_output = last_conv_layer.output

# 3. 최종 도착점: 원본 모델의 최종 출력 (클래스 예측 점수)
model_output = cnn_model.output

# 4. 특징 맵과 최종 출력을 동시에 반환하는 새로운 모델 생성
grad_model = Model(inputs=inputs, outputs=[last_conv_output, model_output])

# 모델 구조 확인
grad_model.summary()

###############
# ==========================================
# 6. Grad-CAM 열화상 이미지 추출 및 시각화
# ==========================================
# 1. 테스트 데이터에서 이미지 하나 선택 (예: 0번째 이미지 - 앵클부츠)
img_index = 0 
test_image = x_test[img_index:img_index+1] # (1, 28, 28, 1) 형태로 차원 유지
true_label = y_test[img_index]

# 2. GradientTape를 사용하여 그래디언트(기울기) 계산
with tf.GradientTape() as tape:
    inputs = tf.cast(test_image, tf.float32)
    
    # grad_model에 이미지를 통과시켜 특징 맵과 예측값을 얻음
    last_conv_output, preds = grad_model(inputs)
    
    # 모델이 가장 높게 예측한 클래스의 인덱스 찾기
    pred_index = tf.argmax(preds[0])
    
    # 타겟 클래스의 예측 점수
    class_channel = preds[:, pred_index]

# 3. 그래디언트 계산: 예측 점수를 마지막 특징 맵으로 미분
grads = tape.gradient(class_channel, last_conv_output)

# 4. 특징 맵에 곱해줄 가중치 계산 (그래디언트의 공간적 평균 - Global Average Pooling)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# 5. 특징 맵의 각 채널에 가중치를 곱하여 하나로 합침
last_conv_output = last_conv_output[0]
heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# 6. ReLU 적용 및 0~1 사이로 정규화 (긍정적인 영향만 남김)
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# 7. 시각화 (열화상 카메라 효과)
plt.figure(figsize=(12, 4))

# ① 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(test_image[0, :, :, 0], cmap='gray')
plt.title(f"Original Image\n(True: {class_names[true_label]})", fontsize=14)
plt.axis('off')

# ② Grad-CAM 히트맵 (열화상)
plt.subplot(1, 3, 2)
# 💡 cmap='jet' 속성이 바로 '열화상 카메라' 스타일의 색상을 만들어줍니다!
plt.imshow(heatmap, cmap='jet') 
plt.title("Grad-CAM Heatmap", fontsize=14)
plt.axis('off')

# ③ 겹쳐진 이미지 (오버레이)
plt.subplot(1, 3, 3)
plt.imshow(test_image[0, :, :, 0], cmap='gray')
# interpolation='bilinear'로 히트맵을 부드럽게 확대해서 원본 이미지 위에 덮어씌웁니다.
plt.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='bilinear') 
plt.title("Superimposed Image", fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()