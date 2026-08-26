import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다
from tensorflow.keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from tensorflow.keras.layers import Dense 
import numpy as np

#keras02.py 와 비교해서 딮러닝 개념이 추가되었다. 고양이 뇌구조 (먹이 ->눈 -> 침) 을 여기서 설명.
# 노드가 뉴런, 파라미터는 y=wx + b할 때 그 계산 할 때 이용하는 것이다.
#1. 데이터 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성 # y =wx + b (w: 가중치, b: 편향)
model = Sequential() # import Sequential 때문에 나왔음
model.add(Dense(4000000, input_dim = 1)) # 앞이 output demension, 뒤가 input_dimension
model.add(Dense(35, input_dim = 40))
model.add(Dense(40, input_dim = 50))
model.add(Dense(50, input_dim = 3000000))
model.add(Dense(3000000, input_dim = 3)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝
model.add(Dense(1, input_dim = 5)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100) #훈련을 시키는데, 그 훈련의 대상이다. x데이터와 y데이터를 훈련 시킨다는 것 근데, 훈련 횟수가 에폭

#4. 평가, 예측
result = model.predict(np.array([7]))
print("7의 예측값 : ", result)

#모든 데이터 분석, 파이토치, 텐서 플로우 모두 위의 순서대로 동일함
