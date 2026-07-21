import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import numpy as np
import pandas as pd #pandas 도 numpy로 구성되어 있다. 
import time 

# keras11_3_kaggle_bike1.py을 카피하였다. 

#1. 데이터 (datetime 포기함, casual, registered도 안씀)
# x는 seanson부터 windspeed까지 사용할 것. y는 count 로 정함.
# 그래서 0번째 컬럼을 인덱스로 할 것이다. 데이터 취급하지 않겠다는 것이다. 그게 바로 index_col=0이다.  인덱스가 세로줄 이다. 

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
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle= True, random_state=11)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620,) (3266,)

# 2 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(8,)))
model.add(Dropout(0.1))
model.add(Dense(10,activation='relu')) #모르면 activation='relu' 활성화 함수 relu 를 사용하면 됨
model.add(Dropout(0.2)) #Dense(10) 밑에 있는 Dropout(0.2): 10개 노드 중 무작위로 2개(20%)의 스위치를 끔
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2)) #Dense(20) 밑에 있는 Dropout(0.2): 앞 층과는 완전히 별개로, 이번 20개 노드 중 무작위로 4개(20%)의 스위치를 따로 끔.
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2)) #만약 Dropout(0.3)이면 이번 30개 노드 중 무작위로 20%의 스위치를 꺼버린다. 
model.add(Dense(35, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(35, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu')) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu')) #초기 가중치가 음수인 것들이 있다. rel
model.add(Dropout(0.2))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(5, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))  
# DropOut은 훈련 때만 적용이 되는 것이다. 실제 예측, 평가에서는 
# DropOut은 모델 자체를 바꾸는게 아니고 훈련 할 때만 랜덤으로 선정한 노드만 훈련에서 제외시킨다. 그래서 fit한 뒤에는 Dropout에서 제외된 모든 노드들이 다시 전부 채워지고, 다시 원래의 완전한 노드가 된다. 거기에 x_test를 넣는 것이다. 
# 이게 빅데이터분석 기사에서 나온다. 

# 3 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time() #time.time하면 현재 시간이 time.time에 저장된다.
model.fit(x_train, y_train, epochs=300, batch_size=32)
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
submit_csv.to_csv(path + "submission_0721_1031.csv") #write라고 생각할 수도 있는데, to_csv이다. csv 너에게 주겠다는 것이다.

print("걸린시간 : ", end_time-start_time,"초") #print("걸린시간 : ", round(end_time-start_time, 2),"초") 이렇게 하면 소수점 둘째 자리까지만 출력한다. 컴활에서도 이방법을 사용한다. 

#CPU 걸린 시간(CUDA를 사용하지 않은 순수): 76.99380779266357 초
#GPU로 실행하면, progress_var도 약간 바뀐다. 무엇으로 실행하고 있는지 구분하기 위해서인 것 같다. 그런데, GPU가 더 느리다. 왜?? GPU가 더 빠르다고 알려져 있는데, 왜 그럴까? 161초걸린다. 
# 단순한 .csv 파일 같이 레이어가 단순한 단순구조는 무조건 CPU가 빠름. CNN, RNN, Transformer 같은 다차원 구조에서는 GPU가 훨씬 빠르다. 
# 머리좋고 힘쎈 적은 수의 인원 VS ㅈㄴ 많은 초딩.  


# rmse= 266.23321533203125 
# rmse= 184.11085510253906