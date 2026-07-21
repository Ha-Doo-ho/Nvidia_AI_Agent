# 기존에는 모델만 가져왔다. 이제는 훈련된 모델을 가져올 것이다. 기존에는 초기 가중치였다.  즉 훈련이 안된 모델을 가져온 것이다. 
# 22_1을 카피했다. 

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes #datasets에는 교육용 데이터가 들어있다.
import numpy as np

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

print(x.shape, y.shape) #(442, 10) (442,) --> 10개고 데이터 수가 많지 않다. batch_size를 줄일 수 있는 기회이다. 

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

model.summary()
#model.save("./_save/kerass22_1_save_model.keras") #가중치를 저장해보자. 확장자는 보통 상관은 없는데, .h5를 하면 된다. 요즘은 .keras도 있다고 한다.  가중치가 들어있긴 한데, 초기 랜덤값이 있다. 그래도 모델이 저장되어 있다고 한다. 
# . 붙이는 것은 상대경로 방식이다. 

#exit()
#R2기준으로 0.6 이상 만들기
# 컴파일 및 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=115, batch_size=4)
#model.save("./_save/kerass22_3_save_model.keras") 여기에 모델을 저장한다. # 상대경로방식 
model.save("c:/nsu_work/_save/keras22_3_save_model.keras") #이건 절대 경로 방식이다. 
 

# 예측 및 평가
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test) #predict는 array 혹은 array_like를 요구한다. 그래서 vector넣어도 array 라서 먹힌다. 
#np.array([]) 이건 리스트라서 어레이가 아니다. np.array() 이건 어레이다. 또한 캘리포니아 데이터셋은 에초에 np.array형태라서 predict에 그냥 넣어도 된다.
print("y_predict", y_predict)

r2 = r2_score(y_test, y_predict)
print("r2: ",r2)

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