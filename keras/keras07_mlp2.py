import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다

from tensorflow.keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from tensorflow.keras.layers import Dense 

import numpy as np


# 보통은 행/열이 뒤바뀌어 있는데, 그때 사용하는 것이 .T 이다.
# 여기서 핵심이 행무시열우선 무조건 암기해야 한다. 열은 input_dim, input_shape, 특성, feature, 속성, attribute
# 무조건 열의 개수를 알면 된다. 그래서 .shape 를 무조건 치는 것이다.add()
# 그래서 국어 영어 수학 점수로 체육 점수를 맞출 수 있는 것이다. 무엇으로? "y의 원값"으로

# 만약 프랑스어 점수도 알 수 있는가? 그렇다. 프랑스어 데이터가 "y"값으로 들어가면 그 y값을 순방향으로 학습시킴.add()
# a의 국영수가 x고 체육, 프랑스어가 y다. b의 국영수가 x고 체육, 프랑스어가 y다. c의 국영수가 x고 체육, 프랑스어가 y다. 
# d의 국영수가 x면 y인 체육, 프랑스어의 점수는 무엇인가? ---> 이걸 알 수 있는 것이다.
# 다시 역방향 올라가면서 w수정시키고, 이제 다시 누군가의 체육, 프랑스어 점수를 물어본다. --> 인공지능은 답을 할 수 있다.


#1. 데이터
# #2개 이상은 [] 추가 
# x = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])  # (2, 6)
x = np.array([[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]])  # (6,2)
x = x.T   #행과 열 바꾸는 함수
#x = x.translate()
print(x.shape)        #(6, 2)
exit()

y = np.array([1,2,3,4,5,6]) # (6,)

#2. 모델구성 # y =wx + b (w: 가중치, b: 편향)
model = Sequential() # import Sequential 때문에 나왔음
# model.add(Dense(200, input_dim = 2)) # 앞이 output demension, 뒤가 input_dimension
model.add(Dense(200, input_shape=(2,)))
model.add(Dense(100))
model.add(Dense(100)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(10)) 
model.add(Dense(1)) 
#훨씬 간결하다. 어차피 두번째 레이어의 아웃풋dim은 3번째 레이어의 인풋 dim이다. 그래서 이렇게 간결하게 쓸 수 있다.
# 즉 상층 레이어의 아웃풋(디멘션)은 하층 레이어(디멘션)의 인풋이다. 그래서 생략이 가능한 것이다. 이전 보다 더욱 간결해진 이유이다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss는 무조건 양수이다. 거리이므로 음수가 될 수 없음. 그래서 양수
model.fit(x, y, epochs=100, ) # x데이터와 y데이터를 훈련 시킨다는 것 근데, 훈련 횟수가 에폭

#4. 평가, 예측 <-- 평가의 기준이 loss율이다. 예측값이 더 가까운 것이 있어도 loss가 더 작
loss = model.evaluate(x,y) #이 loss는 100번 돌리고 난 이후 y' 와 y의 차이가 될 것이다.
print("loss = ", loss)

result = model.predict(np.array([[7,13]]))   #(1,2)
print("7의 예측값 : ", result)




