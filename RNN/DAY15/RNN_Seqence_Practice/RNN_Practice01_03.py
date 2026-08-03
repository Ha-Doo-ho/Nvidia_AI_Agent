import tensorflow as tf 
import pandas as pd 
import numpy as np 
import time 
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, LSTM, Dense, Dropout
from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping

# 1-1 데이터 가져오기 
path = "./_data/"
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

print(test_csv)
#exit()

# 1-2 데이터 전처리 
x = train_csv.drop(['casual','registered','count'],axis=1)
y = train_csv['count']

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=False)

std_scaler = StandardScaler() 
x_train = std_scaler.fit_transform(x_train) #여기서 Pandas가 없어지고, 순수 Numpy로 되돌아간다. 
x_val = std_scaler.transform(x_val)
x_test = std_scaler.transform(test_csv)

print(x_train.shape) #(8708, 8)
#exit()

train_ds = timeseries_dataset_from_array(data=x_train, targets=y_train.values(), sequence_length=24, batch_size=32, shuffle=False)
# data: 문제, target: 그 시간을 주면 결국 그다음은 몇이냐? 즉 답임. sequence_length: 윈도잉할 크기. 즉, 과거 데이터들이다.  
# 만약 과거 데이터가 24개 있어도 현재 데이터가 없으면, 그건 윈도잉 알아서 빼준다 

val_ds = timeseries_dataset_from_array(data=x_val, targets=y_val.values(), sequence_length=24, batch_size=64, shuffle=False)     

# 2 모델 구성
rnn_sequential_model = Sequential()
rnn_sequential_model.add(Input(shape=(24, x_train.shape[1]))) # 윈도우 사이즈, 알고 싶은 열 이 들어간다. 
rnn_sequential_model.add(SimpleRNN(units=64, activation='relu'))
rnn_sequential_model.add(Dropout(0.2))
rnn_sequential_model.add(Dense(32, activation='relu'))
rnn_sequential_model.add(Dense(1))

# 3 컴파일 및 훈련
rnn_sequential_model.compile(optimizer="adam",loss="mse", metrics=['rmsle'])

es = EarlyStopping(monitor="val_loss",patience=100, mode="min", restore_best_weights=True)

# validation_split(X) -> validation_data(O)
start_time = time.time()
rnn_sequential_model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es])
end_time = time.time()

# 평가 및 예측
y_predict = rnn_sequential_model.predict(x_val)
loss, rmsle = rnn_sequential_model.evaluate(x_val, y_val)


