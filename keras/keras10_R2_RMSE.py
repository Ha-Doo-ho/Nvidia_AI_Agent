import tensorflow as tf
print(tf.__version__)  #사용하기 위해 _2개 붙인 것일 뿐이다

from keras.models import Sequential #tensorflow(폴더)의 keras(폴더)의 models(폴더) 의 Sequential클래스
from keras.layers import Dense 
from sklearn.model_selection import train_test_split

import numpy as np

#keras09_scatter2.py와 비교했을 때, RMSE 개념이 추가되었다. 
#이때 추가되는 것이 Accuracy 정확도이다. 
#인공지능 모델은 2가지가 있다. 하나는 회귀 모델, 다른 하나가 분류모델이다. (분류 모델의 지표가 Accuracy)
#회귀에서도 분류가 Accuracy라는 지표를 가지듯이 회귀도 평가 지표가 있다. 그것이 R2스코어이다.
#loss도 있는데, R2는 최대가 1임. R2가 0.7이면, Accuracy가 0.7이라고 생각하면 된다. (일단 이정도만 알면 됨)
# 여기서는 R2_RMSE를 적용시킬 것이다. 

#1. 데이터 (가장 중요한 것은 데이터이다. Garvage in garbage out) 가장 중요한 것은 데이터의 분리이다.

# #2개 이상은 [] 추가   
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15,9, 6, 17, 23,21,20]) 
print(x.shape, y.shape) 

#[실습] train과 test를 섞어서 랜덤하게 7:3을 뽑는다.
# 힌트 사이킷 런
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8,  #test_size=0.3도 된다. 그냥 자기 마음대로 하면 된다. 대신 둘 다 적었을 때 1이 넘으면 에러가 난다. 그리고 1보다 작으면 소중한 데이터가 날아간 상태가 된다.
                                                    shuffle=True,  # Default가 True인데, 가끔 분석할 때 suffle를 False로 둘 때가 있다. 그러나 default값이 True이므로 shuffle=True를 빼고 하는 경우가 일반적이다. 
                                                    random_state=55)
# train_size=0.7 0.7은 70%를 의미. random_state (랜덤 난수) 랜덤값이 아주 좋은 수가 나왔을 때, 그 랜덤 값을 고정시켜주는 것(21번에 해당하는 난수를 주는 것이다.)이다. 그 수를 21로 고정시킨 것이다.
# random_state를 없에면 실행시킬때마다 계속 랜덤하게 뽑아서 테스트 할 때마다 계속 변동이 된다. 일관되게 확인하기 위해서 랜덤 난수를 준다. 랜덤 난수에 해당하는 방법으로 분리해서 섞는다. 그래서 랜덤 난수가 고정되면 늘 숫자는 고정된다.  
# 이 random_state가 모델 향상에 굉장히 중요한 영향을 끼친다. 
print(x_train, x_test)
print(x_train, y_test)

print(x_test, y_test)
print(x_train, y_train)

print(x_train.shape, x_test.shape) #(7,) | (3,)
print(y_train.shape, y_test.shape) #(7,) | (3,) 이건 습관이 되어야 한다.


#2. 모델구성 # y =wx + b (w: 가중치, b: 편향)
model = Sequential() # import Sequential 때문에 나왔음
# model.add(Dense(200, input_dim = 2)) # 앞이 output demension, 뒤가 input_dimension
model.add(Dense(50, input_shape=(1,)))   # data 1덩어리 ([] 1개니까) 집어넣으니까 (1,)이다.
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(200)) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1)) # 답은 마지막에 덩어리 하나만 나오니까  1이다.
#훨씬 간결하다. 어차피 두번째 레이어의 아웃풋dim은 3번째 레이어의 인풋 dim이다. 그래서 이렇게 간결하게 쓸 수 있다.
# 즉 상층 레이어의 아웃풋(디멘션)은 하층 레이어(디멘션)의 인풋이다. 그래서 생략이 가능한 것이다. 이전 보다 더욱 간결해진 이유이다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss는 무조건 양수이다. 거리이므로 음수가 될 수 없음. 그래서 양수
model.fit(x_train, y_train, epochs=100, ) # x데이터와 y데이터를 훈련 시킨다는 것 근데, 훈련 횟수가 에폭

#4. 평가, 예측 <-- 평가의 기준이 loss율이다. 예측값이 더 가까운 것이 있어도 loss가 더 작
loss = model.evaluate(x_test,y_test) #이 loss는 100번 돌리고 난 이후 y' 와 y의 차이가 될 것이다.
print("loss = ", loss) #  221인 경우 loss =  7.653777
                        # 300인 경우 loss =  1.3189462 
                        # 이래서 random_state 가 그래서 중요하다. 다른 것 안바꾸고 random_state 하나 바꿨을 뿐인데, loss값이 확 떨어졌다. 그래서 random_state 즉, 랜덤 난수가 실무에서 매우 중요한 이유이다. 
                        # 220인 경우 loss =  1.0825936
                        
y_predict = model.predict([x_test]) # 100번째 w를 이용해서 y'=wx+b
print("y_test의 원값 : ", y_test)
print("[x_test]의 예측값 : ", y_predict)
###############################################################################################
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
r2 = r2_score(y_test, y_predict) #test와 예측값을 하면 된다.
print(f"r2_score{r2}")

rmse = root_mean_squared_error(y_test, y_predict)
print(f"rmse: {rmse}")

mse = mean_squared_error(y_test, y_predict)
print(f"mse: {mse}")