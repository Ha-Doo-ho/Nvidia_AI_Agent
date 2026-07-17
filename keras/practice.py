import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1 데이터 --> 2개 이상은 리스트
x = np.array([[10,20,30,40,50], 
             [30,40,50,60,70],
             [40,50,60,70,80]])

y = np.array([[60,70],
             [70,80],
             [80,90]])

print(f"x.shape{x.shape}")
print(f"y.shape{y.shape}")

#exit()

# 2 모델 구성
model = Sequential()
model.add(Dense(50,input_shape=(5,)))
model.add(Dense(150))
model.add(Dense(200)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(2)) 

# 3 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

# 4 예측 및 평가
loss = model.evaluate(x,y)
result = model.predict(np.array([[40,50,60,70,80]]))
print(f"loss값 = {loss}")
print(f"result값 = {result}")
