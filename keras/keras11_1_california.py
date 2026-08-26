import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import fetch_california_housing #datasets에는 교육용 데이터가 들어있다.
import numpy as np

#1 데이터
datasets = fetch_california_housing() #데이터 셋 내부에는 캘리포니아 집값 내부에 들어가있다.
#print(datasets)
#print(datasets.DESCR) #조금 더 쉽게 알아볼 수 있다. 그래서 보통 이걸 사용한다.
#print(datasets.feature_names) 빠르게 열에 어떤 데이터 특성이 있는지 알 수 있다. 
print(datasets.feature_names) # 열 이름이다. feature, attrubute, 특징, 열 모두 같은 말이다. 
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

#x,y 자체가 섞여있다. 데이터 셋을 분리해야 한다. sklearn은 x,y를 섞어놨다. 그래서 분리해야 한다. 
x = datasets.data 
y = datasets.target
print("x: ",x)
print("y: ",y)

np.array([])

print(x.shape, y.shape) #(20640, 8) (20640, )

# train, test로 자른다. 
x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=7, shuffle=True)
print(x_train.shape, x_test.shape) #(15480, 8) (5160, 8) 이걸 무조건 해봐야 한다. 실무에서는 반드시 이걸 함으로써 제대로 분리되었는지 확인한다.
print(y_train.shape, y_test.shape) # (15480,) (5160,)
#exit()

# 모델 구성
model = Sequential()
model.add(Dense(10,input_shape=(8,)))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(68))
model.add(Dense(126))
model.add(Dense(32))
model.add(Dense(16)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(8))
model.add(Dense(1)) 


#R2기준으로 0.6 이상 만들기
# 컴파일 및 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=50)

# 예측 및 평가
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test) #predict는 array 혹은 array_like를 요구한다. 그래서 vector넣어도 array 라서 먹힌다. 
#np.array([]) 이건 리스트라서 어레이가 아니다. np.array() 이건 어레이다. 또한 캘리포니아 데이터셋은 에초에 np.array형태라서 predict에 그냥 넣어도 된다.
print("y_predict", y_predict)

r2 = r2_score(y_test, y_predict)
print("r2: ",r2)

"""
    train_size 0.8
    random_state = 7
    
"""