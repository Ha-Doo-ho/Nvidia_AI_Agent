import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다

from tensorflow.keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

import numpy as np

#keras08_train_test1.py와 비교해서 7:3 으로 하면 데이터 불균형이 생길 수 있다. 그래서 무조건 랜덤하게 훈련 데이터를 뽑고 나머지를 테스트 데이터로 만든다. 이건 실무에서 무조건 이렇게 한다. 
#특히 시계열 데이터는 저러한 문제가 가장 크게 나타날 수 있기에, 어차피 랜덤 섞는 것은 기본이지만, 시계열 데이터에서는 그 중요성이 더 강조된다. 
#keras08_train_test1.py와 비교해서 sklearn (사이킷 런) 을 설치해야 한다. 실무에서는 sklearn의 train_test_split로 데이터를 섞어서 불균형을 방지한다.

#1. 데이터
# #2개 이상은 [] 추가   
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,  #test_size=0.3도 된다. 그냥 자기 마음대로 하면 된다. 대신 둘 다 적었을 때 1이 넘으면 에러가 난다. 그리고 1보다 작으면 소중한 데이터가 날아간 상태가 된다.
                                                    shuffle=True,  # Default가 True인데, 가끔 분석할 때 suffle를 False로 둘 때가 있다. 그러나 default값이 True이므로 shuffle=True를 빼고 하는 경우가 일반적이다. 
                                                    random_state=220)
# train_size=0.7 0.7은 70%를 의미. random_state (랜덤 난수) 랜덤값이 아주 좋은 수가 나왔을 때, 그 랜덤 값을 고정시켜주는 것(21번에 해당하는 난수를 주는 것이다.)이다. 그 수를 21로 고정시킨 것이다.
# random_state를 없에면 실행시킬때마다 계속 랜덤하게 뽑아서 테스트 할 때마다 계속 변동이 된다. 일관되게 확인하기 위해서 랜덤 난수를 준다. 랜덤 난수에 해당하는 방법으로 분리해서 섞는다. 그래서 랜덤 난수가 고정되면 늘 숫자는 고정된다.  
# 이 random_state가 모델 향상에 굉장히 중요한 영향을 끼친다. 
print(x_train, x_test)
print(x_train, y_test)

# # [실습] 넘파이 리스트의 슬라이싱 
# x_train = np.array(x[:7]) #혹은 x_train = x[:7]
# y_train = np.array(y[:7])
# #x_train=np.random(x_train)
# print(x_train, y_train)

# x_test = np.array(x[7:]) #혹은 x_test = x[7:]도 됨.
# y_test = np.array(y[7:])
# print(x_test, y_test)

#[실습] train과 test를 섞어서 랜덤하게 7:3을 뽑는다.
# 힌트 사이킷 런
print(x_test, y_test)
print(x_train, y_train)


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
model.add(Dense(100, input_shape=(1,)))   # data 1덩어리 ([] 1개니까) 집어넣으니까 (1,)이다.
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
print("loss = ", loss) #  221인 경우 loss =  7.653777
                        # 300인 경우 loss =  1.3189462 
                        # 이래서 random_state 가 그래서 중요하다. 다른 것 안바꾸고 random_state 하나 바꿨을 뿐인데, loss값이 확 떨어졌다. 그래서 random_state 즉, 랜덤 난수가 실무에서 매우 중요한 이유이다. 
                        # 220인 경우 loss =  1.0825936
result = model.predict(np.array([11]))  
print("11의 예측값 : ", result)
