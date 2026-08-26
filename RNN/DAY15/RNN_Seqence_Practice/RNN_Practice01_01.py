import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import timeseries_dataset_from_array

# 1. 데이터 로드 (경로는 회원님 환경에 맞게 유지)
path = "./_data/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

# 2. X(날씨/환경), y(대여량) 분리
X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

# =======================================================
# 🚨 [변경점 1] 섞지 마세요! (shuffle=False)
# =======================================================
# 과거의 데이터로 미래를 예측해야 하므로 순서를 절대 섞으면 안 됩니다.
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)

# =======================================================
# 🚨 [변경점 2] 스케일링 누수 방지
# =======================================================
# 오직 X(문제지)만 스케일링합니다. y(자전거 대여량)는 그대로 둡니다.
# Train으로 기준(fit)을 잡고, Val과 Test는 변환(transform)만 합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_csv)

# =======================================================
# 🚨 [변경점 3] 2D(표) -> 3D(시계열 윈도우) 변환
# =======================================================
window_size = 24  # 24시간의 흐름을 보겠다
batch_size = 32

# X와 y를 같이 넣어주면 Keras가 알아서 매칭해 줍니다.
# 작동 원리: [0~23시간의 X] -> [23번째 시간의 y] 쌍으로 묶어냄
train_ds = timeseries_dataset_from_array(
    data=X_train_scaled,      # 문제지 (8개의 특성)
    targets=y_train.values,   # 정답지 (대여량 1개)
    sequence_length=window_size,
    batch_size=batch_size
)

val_ds = timeseries_dataset_from_array(
    data=X_val_scaled,
    targets=y_val.values,
    sequence_length=window_size,
    batch_size=batch_size
)

# 4. 모델 구성
model = Sequential(name="Kaggle_Bike_RNN")
# 입력 형태: (시간의 길이 24, 특성 개수 8)
model.add(Input(shape=(window_size, X_train.shape[1]))) 
model.add(SimpleRNN(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) # 회귀(수치 예측)이므로 1개 출력

# 5. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# validation_split(X) -> validation_data(O)
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es])