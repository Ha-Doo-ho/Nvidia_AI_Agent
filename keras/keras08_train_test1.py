import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다

from tensorflow.keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from tensorflow.keras.layers import Dense 

import numpy as np

# 지금 전부 훈련데이터로만 훈련하고 평가 예측도 훈련데이터를 그대로 사용했다.
# 이건 당연히 잘 나올 수 밖에 없다. 잘 나와야 정상이다. --> 오버핏이다.
# 이제와서 데이터를 추가하는 것은 불가능하다. (ㅈㄴ어려움) 그래서 데이터를 훈련데이터(7), 테스트데이터(3 보통 그렇다.)로 자른다.
# 물론 재활용은 가능하다. (이건 나중에)
# 평가가 훈련보다 좋을 수는 없다. 
# Accuraccy: 정확도로 0~1.0 
# 이건 원칙이다 어떤 데이터를 받으면 training | test로 분리

# 이제 이것을 코드로 적용해 보자.

#1. 데이터
# #2개 이상은 [] 추가 
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train = np.array([1,2,3,4,5,6,7,]) # 끝에 , 가 있다. 이 의미는 더 넣을 수 있을 가능성이 있다는 뜻이다. 파이썬 기본문법이다. 
y_train = np.array([1,2,3,4,5,6,7,])
.txt .DCM --> .PNG
x_test = np.array([8,9,10,])
y_test = np.array([8,9,10,])

print(x_train.shape, x_test.shape) #(7,) | (3,)
print(y_train.shape, y_test.shape) #(7,) | (3,) 이건 습관이 되어야 한다.

# print(x.shape) #(10, )   
# print("x",x) 
# # exit()

# print(y.shape)  #(10, )
# print("y",y)
# #exit()


#2. 모델구성 # y =wx + b (w: 가중치, b: 편향)
model = Sequential() # import Sequential 때문에 나왔음
# model.add(Dense(200, input_dim = 2)) # 앞이 output demension, 뒤가 input_dimension
model.add(Dense(100, input_shape=(1,)))   # data 1덩어리 집어넣으니까 (1,)이다.
model.add(Dense(150))
model.add(Dense(200)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1)) # 답은 마지막에 덩어리 하나만 나오니까  1이다.
#훨씬 간결하다. 어차피 두번째 레이어의 아웃풋dim은 3번째 레이어의 인풋 dim이다. 그래서 이렇게 간결하게 쓸 수 있다.
# 즉 상층 레이어의 아웃풋(디멘션)은 하층 레이어(디멘션)의 인풋이다. 그래서 생략이 가능한 것이다. 이전 보다 더욱 간결해진 이유이다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss는 무조건 양수이다. 거리이므로 음수가 될 수 없음. 그래서 양수
model.fit(x_train, y_train, epochs=256, ) # x데이터와 y데이터를 훈련 시킨다는 것 근데, 훈련 횟수가 에폭

#4. 평가, 예측 <-- 평가의 기준이 loss율이다. 예측값이 더 가까운 것이 있어도 loss가 더 작
loss = model.evaluate(x_test,y_test) #이 loss는 100번 돌리고 난 이후 y' 와 y의 차이가 될 것이다.
print("loss = ", loss)


result = model.predict(np.array([11]))  
print("11의 예측값 : ", result)




