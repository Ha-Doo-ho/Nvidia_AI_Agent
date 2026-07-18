import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import numpy as np
import pandas as pd #pandas 도 numpy로 구성되어 있다. 
import time 
from keras.callbacks import EarlyStopping #keras13_validation1.py와 비교해서 이게 추가되었다. 
                                          #callbacks에 좋은 게 많이 있다고 한다. 


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

# 데이터를 섞은 뒤, train과 test로 나눈다. train_test_split를 사용한다. 이건 필수다.
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle= True, random_state=10)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620,) (3266,)

# 2 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(8,)))
model.add(Dense(10,activation='relu')) #모르면 activation='relu' 활성화 함수 relu 를 사용하면 ㅗ딤
model.add(Dense(15, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu')) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(20, activation='relu')) #초기 가중치가 음수인 것들이 있다. rel
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu')) 
model.add(Dense(5, activation='relu')) 
model.add(Dense(1, activation='relu'))  #이 activation도 default가 있는데, activation='linear'이다. y=wx+b 인데, 통상적으로 relu가 더 좋다. (오히려 안좋아지는 경우도 있다.)
#그래서 성능 안나오면 relu 를 사용하면 된다. 그래서 이것도 하이퍼 파라미터 튜닝이다. 

# 3 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
es= EarlyStopping(monitor='val_loss',mode='min', patience=100, restore_best_weights=True,) #나는 가장 좋은 값을 가중치로 갔겠다. 
#restore_best_weights: 학습이 종료된 후 가장 성능이 좋았던 epoch의 가중치로 되돌릴지를 결정

start_time = time.time() #time.time하면 현재 시간이 time.time에 저장된다.
model.fit(x_train, y_train, epochs=10000000, batch_size=8, validation_split=0.2,callbacks=[es]) #EarlyStopping형태 callbacks에 들어갈 게 es 말고 더 있으니까 리스트 (2개 이상은 리스트)
end_time = time.time()

# 4 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

rmse = root_mean_squared_error(y_test,y_predict) #rmse= 155.07630920410156
print("loss = ", loss)
print("y_predict", y_predict)
print("rmse=",rmse)


############ CSV 파일 만들기 #################
y_submit = model.predict(test_csv) 
#print(y_submit)
submit_csv['count'] = y_submit #이제 답지의 count에 y_submit을 넣는다.
print(submit_csv)
submit_csv.to_csv(path + "submission_0717_1453.csv") #write라고 생각할 수도 있는데, to_csv이다. csv 너에게 주겠다는 것이다.

print("걸린시간 : ", end_time-start_time,"초") #print("걸린시간 : ", round(end_time-start_time, 2),"초") 이렇게 하면 소수점 둘째 자리까지만 출력한다. 컴활에서도 이방법을 사용한다. 
