import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다

from tensorflow.keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from tensorflow.keras.layers import Dense 

import numpy as np


#모델 평가를 하기 위한 evaluate.py 이다. keras03_deep2.py 와 비교해서 정말 평가를 하기 위해 loss를 사용했다.
# loss는 0이 베스트이다. 이건 미래를 완벽하게 예측했다는 것이다.

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성 # y =wx + b (w: 가중치, b: 편향)
model = Sequential() # import Sequential 때문에 나왔음
model.add(Dense(50, input_dim = 1)) # 앞이 output demension, 뒤가 input_dimension
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(70)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(60)) 
model.add(Dense(1)) 
#훨씬 간결하다. 어차피 두번째 레이어의 아웃풋dim은 3번째 레이어의 인풋 dim이다. 그래서 이렇게 간결하게 쓸 수 있다.
# 즉 상층 레이어의 아웃풋(디멘션)은 하층 레이어(디멘션)의 인풋이다. 그래서 생략이 가능한 것이다. 이전 보다 더욱 간결해진 이유이다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss는 무조건 양수이다. 거리이므로 음수가 될 수 없음. 그래서 양수
model.fit(x, y, epochs=100) #훈련을 시키는데, 그 훈련의 대상이다. x데이터와 y데이터를 훈련 시킨다는 것 근데, 훈련 횟수가 에폭
# 100번째 w 값이 나오게 된다.

#4. 평가, 예측 <-- 평가의 기준이 loss율이다. 예측값이 더 가까운 것이 있어도 loss가 더 작
loss = model.evaluate(x,y) #이 loss는 100번 돌리고 난 이후 y' 와 y의 차이가 될 것이다.
print("loss = ", loss)

result = model.predict(np.array([7]))
print("7의 예측값 : ", result)

#모든 데이터 분석, 파이토치, 텐서 플로우 모두 위의 순서대로 동일함
