import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다
from tensorflow.keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from tensorflow.keras.layers import Dense 
import numpy as np

#1. 데이터 (keras01.py와 비교해서, 데이터 양이 늘어났다.)
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성 # y =wx + b (w: 가중치, b: 편향)
model = Sequential() # import Sequential 때문에 나왔음
model.add(Dense(1, input_dim = 1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=600) #훈련을 시키는데, 그 훈련의 대상이다. x데이터와 y데이터를 훈련 시킨다는 것 근데, 훈련 횟수가 에폭

#4. 평가, 예측
result = model.predict(np.array([7]))
print("7의 예측값 : ", result)

#모든 데이터 분석, 파이토치, 텐서 플로우 모두 위의 순서대로 동일함.

#결과 성능이 향상되었다. (무조건 향상 되는 것은 아니다.)
