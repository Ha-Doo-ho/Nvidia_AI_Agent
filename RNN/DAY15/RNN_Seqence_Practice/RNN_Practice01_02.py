import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import timeseries_dataset_from_array

# ---------------------------------------------------------
# 1. Data (sklearn을 통한 자전거 대여 시계열 데이터 로드)
# ---------------------------------------------------------
print("데이터를 다운로드 중입니다. (인터넷 환경에 따라 수 초 소요)...")
# OpenML 서버에서 자전거 대여 수요 데이터를 직접 불러옵니다.
bike_data = fetch_openml(name='bike_sharing', parser='auto')

# 여러 특성 중 정답(Target)인 '총 대여량(count)'만 추출하여 시계열로 사용합니다.
# 1차원 형태를 스케일러와 RNN에 넣기 위해 2차원(N, 1)으로 바꿉니다.
series_data = bike_data.target.values.reshape(-1, 1).astype('float32')

# 실습을 빠르게 하기 위해 최신 10,000시간의 데이터만 잘라서 사용합니다.
series_data = series_data[-10000:]

# 1-1. 데이터를 섞지 않고 시간 순서대로 70% / 15% / 15% 분리 (shuffle=False 유지!)
train_data, temp_data = train_test_split(series_data, test_size=0.3, shuffle=False)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

print(f"데이터 길이 - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# 1-2. 스케일링 (Data Leakage 방지를 위해 Train 기준으로만 fit!)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# 1-3. Keras 내장 메서드로 Windowing (각 구간별로 분리해서 파이프라인 생성)
window_size = 24  # 과거 24시간의 대여량 흐름을 보고
batch_size = 32   # 바로 다음 1시간의 대여량을 예측

train_ds = timeseries_dataset_from_array(
    data=train_scaled[:-1],             # 문제. 몇 일치가 들어온다. 
    targets=train_scaled[window_size:], # 답. 몇 일치의 데이터가 이러한데, 답은 과연? 
    sequence_length=window_size,        #  윈도우 사이즈. 
    batch_size=batch_size               # 배치 사이즈. 
)

val_ds = timeseries_dataset_from_array(
    data=val_scaled[:-1],
    targets=val_scaled[window_size:],
    sequence_length=window_size,
    batch_size=batch_size
)

test_ds = timeseries_dataset_from_array(
    data=test_scaled[:-1],
    targets=test_scaled[window_size:],
    sequence_length=window_size,
    batch_size=batch_size
)

# ---------------------------------------------------------
# 2. Model Configuration
# ---------------------------------------------------------
rnn_model = Sequential(name="rnn_bike_predictor")
rnn_model.add(Input(shape=(24, 1))) # (Timesteps=24, Features=1)
rnn_model.add(SimpleRNN(units=64, activation='tanh'))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(1)) 

# ---------------------------------------------------------
# 3. Compile and Fit
# ---------------------------------------------------------
rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)

start_time = time.time()
rnn_model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[es])
end_time = time.time()

# ---------------------------------------------------------
# 4. Evaluate and Predict
# ---------------------------------------------------------
loss, mae = rnn_model.evaluate(test_ds)
print(f"\n최종 결과 -> loss(MSE): {round(loss, 4)}, MAE: {round(mae, 4)}")
print(f"걸린시간 : {round(end_time - start_time, 2)}초")

# 시각화 (테스트 데이터 1개 배치만 꺼내서 예측값과 실제값 비교)
for x_batch, y_batch in test_ds.take(1):
    y_predict_batch = rnn_model.predict(x_batch)
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_batch.numpy(), label='Actual Bike Count', marker='o', linestyle='-') 
    plt.plot(y_predict_batch, label='Predicted Bike Count', marker='x', linestyle='--')
    plt.title("Bike Demand Prediction (Past 24h -> Next 1h)")
    plt.ylabel("Scaled Demand")
    plt.xlabel("Time Step (Hours)")
    plt.legend()
    plt.grid(True)
    plt.show()