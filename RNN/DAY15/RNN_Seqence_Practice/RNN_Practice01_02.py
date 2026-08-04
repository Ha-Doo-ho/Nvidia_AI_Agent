import tensorflow as tf 
import pandas as pd 
import numpy as np 
import time 
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, LSTM, Dense, Dropout
from keras.utils import timeseries_dataset_from_array #이걸 validation_split이 지원하지 않음. kearas임에도 불구하고 
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


#print(x_train.shape) #(8708, 8)
print(x_train) #찍어보면 fit_transform 때문에, 순수 Numpy형태, 즉 값 만 있는 형태라는 것을 알 수 있다. 
exit()

train_ds = timeseries_dataset_from_array(data=x_train, targets=y_train.values, sequence_length=24, batch_size=32, shuffle=False)
# data: 문제, target: 그 시간을 주면 결국 그다음은 몇이냐? 즉 답임. sequence_length: 윈도잉할 크기. 즉, 과거 데이터들이다.  
# 만약 과거 데이터가 24개 있어도 현재 데이터가 없으면, 그건 윈도잉 알아서 빼준다 
# .values하는 이유: y_train에는 아직 인덱스와 컬럼이 덕지덕지 남아 있음. 그것까지 없에고 순수하게 값으로만 x_train과 비교해야 함. 인덱스와 컬럼이 있으니까 값만 있는 x_train이랑 비교를 못함. 
# 그래서 .values로 인덱스, 컬럼 다 지우고 순수한 값만 남기는 것이다. 
# 더 직관적인 메서드가 있는데, 그게 to_numpy()이다. 값만 남긴다. 

val_ds = timeseries_dataset_from_array(data=x_val, targets=y_val.values, sequence_length=24, batch_size=64, shuffle=False)     
#timeseries_dataset_from_array 통과하면 무조건 3차원 됨. 이건 외워야 함. 아니면, .element_spec메서드로 알아야 함. 

print(train_ds.element_spec, val_ds.element_spec) 
#(TensorSpec(shape=(None, None, 8), dtype=tf.float64, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))
#(TensorSpec(shape=(None, None, 8), dtype=tf.float64, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))
#timeseries_dataset_from_array를 통과한 데이터 셋은 ★★3차원으로 변한다.★★ 데이터번호(배치사이즈임), 시간, 차원 순서로 변한다는 것이다. 

#print("......",type(val_ds)) #'tensorflow.python.data.ops.batch_op._BatchDataset' tf.data.Dataset이다. 
#exit()

# 2 모델 구성
rnn_sequential_model = Sequential()
rnn_sequential_model.add(Input(shape=(24, x_train.shape[1]))) # 윈도우 사이즈, 알고 싶은 열 이 들어간다. 
rnn_sequential_model.add(SimpleRNN(units=64, activation='relu'))
rnn_sequential_model.add(Dropout(0.2))
rnn_sequential_model.add(Dense(32, activation='relu'))
rnn_sequential_model.add(Dense(1))

# 3 컴파일 및 훈련
rnn_sequential_model.compile(optimizer="adam",loss="mse", metrics=['mae'])

es = EarlyStopping(monitor="val_loss",patience=100, mode="min", restore_best_weights=True)

# validation_split(X) -> validation_data(O) validation_split은 알아서 해주는데 순서에 민감하거나 정렬되어 있는 다음과 같은 상황에서는 validation_data에 넣어야 한다. 
# validation_split과 validation_data는 둘 다 validation용도로 사용하는데, 결국 선택이다. 자동으로 하길 원하면 validation_split이고, 순서가 중요해서 내가 직접 만든경우는 validation_data 
# tf.data.Dataset이나 제너레이터를 사용할 때는 validation_split 대신 validation_data 인자에 직접 검증 셋 튜플 (x_val, y_val)이나 데이터셋을 넘겨주어야 한다. 
# validation_split은 데이터가 배열/텐서일 때만 사용 가능하다. 
start_time = time.time()
rnn_sequential_model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es])
end_time = time.time()

# 평가 및 예측
y_predict = rnn_sequential_model.predict(val_ds)
loss, mae = rnn_sequential_model.evaluate(val_ds) #mae가 loss이긴 하다. mae의 e가 error인데, loss, error, cost, 비용 전부 다 같은 말. 

print("mae: ",mae)


