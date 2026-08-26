import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes #datasets에는 교육용 데이터가 들어있다.
import numpy as np
import time

#11-2 카피 하였다.
# 사람에게 훈련 정도를 보여주기 위해, 딜레이를 걸어놓은 것이다. 이걸 1에폭으로 하니 딜레이가 더 걸린다. 
# 어차피 훈련하는 것 아는데, 굳이 볼 필요가 없지 않을까? 그러면 속도가 더 빨라지지 않을까? --> 이 아이디어가 추가된 것아다.  --> 그 아이디어가 verbose에 적용되어 있다. 
#keras11_1_california.py는 캘리포니아 집값이다. 이번엔, 당뇨병이다. 

#1 데이터
datasets = load_diabetes()
#print(datasets)
#print(datasets.DESCR) #조금 더 쉽게 알아볼 수 있다. 그래서 보통 이걸 사용한다.
#print(datasets.feature_names) 빠르게 열에 어떤 데이터 특성이 있는지 알 수 있다. 
print(datasets.feature_names) # 열 이름이다. feature, attrubute, 특징, 열 모두 같은 말이다. 
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

#exit()

#x,y 자체가 섞여있다. 데이터 셋을 분리해야 한다. sklearn은 x,y를 섞어놨다. 그래서 데이터 자체를 분리해야 한다. 
x = datasets.data 
y = datasets.target
print("x: ",x)
print("y: ",y)

print(x.shape, y.shape) #(442, 10) (442,) --> 열개수 10개고 데이터 수가 많지 않다. batch_size를 줄일 수 있는 기회이다. 

# train, test로 자른다. 
x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=11, shuffle=True)
print(x_train.shape, x_test.shape) #(15480, 8) (5160, 8) 이걸 무조건 해봐야 한다. 실무에서는 반드시 이걸 함으로써 제대로 분리되었는지 확인한다.
print(y_train.shape, y_test.shape) # (15480,) (5160,) shpe는 무조건 찍어야 한다. 그래야 열의 개수를 알 있고, 그 열의 개수를 통해, input_dim과 최종 output_dim을 결정할 수 있다. 
#exit()

# 모델 구성
model = Sequential()
model.add(Dense(10,input_shape=(10,)))
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


#R2기준으로 0.6 이상 만들기
# 컴파일 및 훈련
model.compile(loss='mse',optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=115, batch_size=32, verbose=72) #vervose = 0이라고 하면 훈련과정인 초록 바가 안나온다. 이 초록바를 사람에게 보여야 하기 때문에, 시간이 조금씩 딜레이 된다. 이 초록바를 안보이게 하면 속도가 더 빨라진다.
end_time = time.time()                                            #default 가 그러면 존재하는 것을 알 수 있다. 그게 verbose = 1이다.  verbose=2도 있는데, 에폭만 본다. 2부터 어떤 숫자를 넣든 에폭만 나온다. 
                                                                  # 성능과 속도의 트레이드오프를 고려해야 할 것이 있다. 그래서 이런 것 까지도 고려해야 할 때(보통 데이터 셋 커지면 이거도 2이상으로 두거나 0으로 둔다.) verbose옵션을 줄 수 있다. 

# 예측 및 평가
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test) #predict는 array 혹은 array_like를 요구한다. 그래서 vector넣어도 array 라서 먹힌다. 
#np.array([]) 이건 리스트라서 어레이가 아니다. np.array() 이건 어레이다. 또한 캘리포니아 데이터셋은 에초에 np.array형태라서 predict에 그냥 넣어도 된다.
print("y_predict", y_predict)

r2 = r2_score(y_test, y_predict)
print("r2: ",r2)

print("걸린 시간 = ",round(end_time - start_time, 2), "초")

"""
    train_size 0.8
    random_state = 10
    에폭: 100
    배치 사이즈: 8
    
    배치 사이즈: 2
    0.51 
    
    배치 사이즈: 4
    0.52
    랜덤 난수: 10
    
    훈련 사이즈: 0.8
    배치 사이즈: 4
    랜덤 난수: 11
    0.54 
    
    훈련 사이즈: 0.85
    랜덤 난수: 11
    에폭: 100
    배치 사이즈: 4
    0.57
    
    훈련 사이즈 : 0.8
    랜덤 난수: 11
    에폭: 110
    배치 사이즈:4
    0.57
"""