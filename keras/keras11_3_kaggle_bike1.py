import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes #datasets에는 교육용 데이터가 들어있다.
import numpy as np
import pandas as pd #pandas 도 numpy로 구성되어 있다. 

#1. 데이터 (datetime 포기함, casual, registered도 안씀)
# x는 seanson부터 windspeed까지 사용할 것. y는 count 로 정함.
# 그래서 0번째 컬럼을 인덱스로 할 것이다. 데이터 취급하지 않겠다는 것이다. 그게 바로 index_col=0이다.  인덱스가 세로줄 이다. 
# keras11_2_diabetes.py 와 비교해서 이제 csv파일, 즉 keggle을 진행해 보자. 
path = "./_data/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) # "./_data/test.csv"
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
#print(train_csv)
# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)
# print(submit_csv)
# print(submit_csv.shape) #(6493, 1)

print(train_csv.columns) #ㅈㄴ 많이 씀 무조건 알아야 함. csv파일에서 컬럼 이름을 확인하려면 .columns를 해야 한다. 

# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='str'
# x(season~windspeed)와 y(count) 우선 x에서는 3개 빼야 한다. register, casual, count(y로 쓸 것)
x = train_csv.drop(['casual','registered', 'count'], axis=1) #3개 뺀다. 여기서도 두개 이상은 리스트 라는 법칙이 적용됨 근데, 열 3개를 뺄 것이니까 axis=1이다. 
print(x)

print("===================================")
y = train_csv['count']
print(y)
print(y.shape) #(10886,) 가 나온다. 인덱스는 데이터로 안친다. 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle= True, random_state=42)

# 2 모델구성
model = Sequential()
model.add(Dense(10,input_shape=(8,)))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(68))
model.add(Dense(126))
model.add(Dense(258))
model.add(Dense(126))
model.add(Dense(32))
model.add(Dense(16)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1)) 

# 3 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2)

# 4 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

rmse = root_mean_squared_error(y_test,y_predict)
print("loss = ", loss)
print("y_predict", y_predict)
print("rmse=",rmse)