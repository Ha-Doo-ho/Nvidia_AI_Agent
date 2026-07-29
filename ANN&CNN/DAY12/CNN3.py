import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense
from keras.callbacks import EarlyStopping
from keras.datasets import fashion_mnist
from keras.layers import GlobalAveragePooling2D

# ==========================================
# 1. 데이터
# ==========================================
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("원본 x_train shape:", x_train.shape)   # (60000, 28, 28)
print("원본 x_test shape :", x_test.shape)    # (10000, 28, 28)
print("y_train shape:", y_train.shape)        # (60000,)
print("y_test shape :", y_test.shape)         # (10000,)

# CNN 입력에 필요한 흑백 채널 1개 추가
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 픽셀값 0~255를 0~1로 정규화
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("전처리 x_train shape:", x_train.shape)  # (60000, 28, 28, 1)
print("전처리 x_test shape :", x_test.shape)   # (10000, 28, 28, 1)


# ==========================================
# 2. Dense와 Conv2D 파라미터 비교
# ==========================================
# Dense: (입력 개수 + bias 1개) * 출력 노드 수
# 28 * 28개의 픽셀을 Dense 64개에 연결
dense_params = (28 * 28 + 1) * 64

# Conv2D: (커널 높이 * 커널 너비 * 입력 채널 + bias 1개) * 필터 수
# 3 * 3 커널, 입력 채널 1개, 필터 64개
conv_params = (3 * 3 * 1 + 1) * 64

print("Dense 파라미터 수:", dense_params)
print("Conv2D 파라미터 수:", conv_params)


# ==========================================
# 3. Convolution 반응값과 ReLU
# ==========================================
patch = np.array([
    [2, 0, 1],
    [1, 3, 0],
    [0, 2, 4]
], dtype=np.float32)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

# 같은 위치끼리 곱한 후 모두 더함
response = float((patch * kernel).sum())

# ReLU: 음수는 0, 양수는 그대로
relu_response = max(0.0, response)

print("Convolution 반응값:", response)
print("ReLU 결과:", relu_response)


# ==========================================
# 4. Conv 출력 크기와 수용 영역
# ==========================================
def conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

size_stride_1 = conv_output_size(28, 3, 0, 1)
size_stride_2 = conv_output_size(28, 3, 0, 2)

# 3x3 Conv를 stride=1로 3번 통과했을 때 수용 영역
receptive_field = 1
for _ in range(3):
    receptive_field += 3 - 1

print("stride=1 출력 크기:", size_stride_1)
print("stride=2 출력 크기:", size_stride_2)
print("Conv 3층 수용 영역:", receptive_field)


# ==========================================
# 5. Flatten과 GlobalAveragePooling2D 비교
# ==========================================
feature_tensor = tf.zeros((2, 7, 7, 64))

flatten_model = Sequential()
flatten_model.add(Input(shape=(7, 7, 64)))
flatten_model.add(Flatten())
flattened = flatten_model(feature_tensor)

gap_model = Sequential()
gap_model.add(Input(shape=(7, 7, 64)))
gap_model.add(GlobalAveragePooling2D())
gap_vector = gap_model(feature_tensor)

flatten_dense_params = (int(flattened.shape[-1]) + 1) * 64
gap_dense_params = (int(gap_vector.shape[-1]) + 1) * 64

print("Flatten 결과 shape:", flattened.shape)
print("GAP 결과 shape:", gap_vector.shape)
print("Flatten -> Dense(64) 파라미터:", flatten_dense_params)
print("GAP -> Dense(64) 파라미터:", gap_dense_params)


# ==========================================
# 6. ANN 모델
# ==========================================
ann_model = Sequential(name="ann_baseline")
ann_model.add(Input(shape=(28, 28, 1), name="Input_layer"))
ann_model.add(Flatten(name="flatten"))
ann_model.add(Dense(64, activation="relu", name="hidden"))
ann_model.add(Dense(10, activation="softmax", name="output"))


# ==========================================
# 7. CNN-valid 모델
# ==========================================
cnn_valid_model = Sequential(name="cnn_valid")
cnn_valid_model.add(Input(shape=(28, 28, 1), name="Input_layer"))
cnn_valid_model.add(Conv2D(32, 3, padding="valid", activation="relu", name="conv_1"))
cnn_valid_model.add(MaxPool2D(pool_size=2, name="pool_1"))
cnn_valid_model.add(Conv2D(64, 3, padding="valid", activation="relu", name="conv_2"))
cnn_valid_model.add(Flatten(name="flatten"))
cnn_valid_model.add(Dense(64, activation="relu", name="hidden"))
cnn_valid_model.add(Dense(10, activation="softmax", name="output"))


# ==========================================
# 8. CNN-same 모델
# ==========================================
cnn_same_model = Sequential(name="cnn_same")
cnn_same_model.add(Input(shape=(28, 28, 1), name="Input_layer"))
cnn_same_model.add(Conv2D(32, 3, padding="same", activation="relu", name="conv_1"))
cnn_same_model.add(MaxPool2D(pool_size=2, name="pool_1"))
cnn_same_model.add(Conv2D(64, 3, padding="same", activation="relu", name="conv_2"))
cnn_same_model.add(Flatten(name="flatten"))
cnn_same_model.add(Dense(64, activation="relu", name="hidden"))
cnn_same_model.add(Dense(10, activation="softmax", name="output"))

print("ANN 파라미터:", ann_model.count_params())
print("CNN-valid 파라미터:", cnn_valid_model.count_params())
print("CNN-same 파라미터:", cnn_same_model.count_params())


# ==========================================
# 9. 컴파일
# ==========================================
ann_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

cnn_valid_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

cnn_same_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# epochs=10이므로 patience는 3으로 설정
es_ann = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    restore_best_weights=True
)

es_valid = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    restore_best_weights=True
)

es_same = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# 10. 세 모델 학습
# ==========================================
# 같은 x_train의 마지막 20%를 validation으로 사용
ann_history = ann_model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es_ann],
    verbose=1
)

valid_history = cnn_valid_model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es_valid],
    verbose=1
)

same_history = cnn_same_model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es_same],
    verbose=1
)


# ==========================================
# 11. Validation 정확도 비교
# ==========================================
ann_best_val_accuracy = np.max(ann_history.history["val_accuracy"])
valid_best_val_accuracy = np.max(valid_history.history["val_accuracy"])
same_best_val_accuracy = np.max(same_history.history["val_accuracy"])

print("ANN 최고 Validation Accuracy:", ann_best_val_accuracy)
print("CNN-valid 최고 Validation Accuracy:", valid_best_val_accuracy)
print("CNN-same 최고 Validation Accuracy:", same_best_val_accuracy)

# CNN-valid와 CNN-same 중 최종 모델 선택
accuracy_gap = abs(valid_best_val_accuracy - same_best_val_accuracy)

if accuracy_gap > 0.001:
    if valid_best_val_accuracy > same_best_val_accuracy:
        selected_model = cnn_valid_model
        selected_name = "CNN-valid"
    else:
        selected_model = cnn_same_model
        selected_name = "CNN-same"
else:
    # 정확도가 거의 같으면 파라미터가 적은 모델 선택
    if cnn_valid_model.count_params() < cnn_same_model.count_params():
        selected_model = cnn_valid_model
        selected_name = "CNN-valid"
    else:
        selected_model = cnn_same_model
        selected_name = "CNN-same"

print("두 CNN 정확도 차이:", accuracy_gap)
print("최종 선택 모델:", selected_name)


# ==========================================
# 12. 최종 모델 평가와 예측
# ==========================================
loss, accuracy = selected_model.evaluate(x_test, y_test)

print(f"최종 Test Loss: {loss:.4f}")
print(f"최종 Test Accuracy: {accuracy:.4f}")

y_predict = selected_model.predict(x_test)
y_predict_classes = np.argmax(y_predict, axis=1)
confidence = np.max(y_predict, axis=1)

print("실제 정답 10개:", y_test[:10])
print("모델 예측 10개:", y_predict_classes[:10])
print("예측 확신도 10개:", np.round(confidence[:10], 3))


# ==========================================
# 13. 혼동행렬
# ==========================================
# sklearn을 추가하지 않고 TensorFlow로 혼동행렬 계산
cm = tf.math.confusion_matrix(
    y_test,
    y_predict_classes,
    num_classes=10
)

print("혼동행렬:")
print(cm.numpy())


# ==========================================
# 14. 확신도가 높은 오답 8개
# ==========================================
wrong_indices = np.flatnonzero(y_predict_classes != y_test)
wrong_order = np.argsort(confidence[wrong_indices])[::-1]
top_wrong_indices = wrong_indices[wrong_order[:8]]

class_names = np.array([
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
])

print("확신도가 높은 오답 8개")
for index in top_wrong_indices:
    print(
        "index:", index,
        "| 실제:", class_names[y_test[index]],
        "| 예측:", class_names[y_predict_classes[index]],
        "| 확신도:", round(float(confidence[index]), 4)
    )


# ==========================================
# 15. 모델 구조 확인
# ==========================================
print("\nANN 모델")
ann_model.summary()

print("\nCNN-valid 모델")
cnn_valid_model.summary()

print("\nCNN-same 모델")
cnn_same_model.summary()
