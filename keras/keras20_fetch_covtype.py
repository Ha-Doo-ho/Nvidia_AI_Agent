#sklearn에서 제공하는 50만개짜리 데이터 셋이다. 
# keras18_softmax_wine.py 를 카피하였다. 

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score #이진 분류 혹은 다중 분류
import pandas as pd
import time 


#1 데이터
datasets = load_digits()
#print(datasets)
#exit()
print(datasets.DESCR) #조금 더 쉽게 알아볼 수 있다. 그래서 보통 이걸 사용한다. :Number of Instances: 150 (50 in each of three classes), :Number of Attributes: 4 numeric --> 열/속성/feature/ --> 행무시,열우선  
#exit() # instance는 행이다.                                                

print(datasets.feature_names) # 빠르게 열에 어떤 데이터 특성이 있는지 알 수 있다.  열 이름이다. feature, attrubute, 특징, 열 모두 같은 말이다. 
#exit()

#x,y 자체가 섞여있다. 데이터 셋을 분리해야 한다. sklearn은 x,y를 섞어놨다. 그래서 데이터 자체를 분리해야 한다. 
# 실무에서는 .data, .target 안쓰고 pandas로 read_csv, to_csv를 사용해야 한다.  
x = datasets.data 
y = datasets.target
print("x: ",x)
print("y: ",y)

#exit()
print(x.shape, y.shape) #(1797, 64) | (1797,) --> 1797개의 0부터 9까지의 데이터  
#exit()
print(np.unique(y, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) 이거 보고 다중분류라는 것을 바로 떠올리고 원핫을 해주어야 한다. 데이터 개수가 상당이 균형적이다. 데이터는 이렇게 균형적인 분포를 보여야 한다. 만약 불균형하다? 데이터를 수집 더 하거나, 증폭해야 함. 
#exit()
########## onehot 01 판다스 이용 ###########
y = pd.get_dummies(y) # 이렇게 하는 이유가 [0, 1, 2 . . .] 이렇게 분류된다. 즉 1차형 범주형으로 된다. 이렇게 되면 너구리인 2가 고양이 1의 2배인가? 라고 생각할 수 있으니까 원-핫 인코딩 하는 것이다.  
print(y)#IT에서 true:1, false: 0이니까 당연한 것이다.
#exit()  

# train, test로 자른다. 
x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=11, shuffle=True)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape) 

#여기서 잘라준 내용에 0과 1만 있으면 ㅈㄴ 문제가 됨. 2를 연습을 못했기 때문이다. 그래서 np.unique를 보는 것이다. 
print(np.unique(y_train, return_counts=True)) # 그런데, 이건 그걸 이걸 만든 사람도 알기 때문에, 다중 분류를 걸면, 자기들이 알아서, 0을 0.75로 훈련, 나머지 0.25를 테스트 1도 0.75로 훈련, 나머지 0.25를 테스트로 잡고 2도 마찬가지이다. 그걸 프린트 해본 것이다. 
#exit()

# 모델 구성
model = Sequential()
model.add(Dense(10,input_shape=(64,), ))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu')) # 여기서는 y=wx+b의 형태로 계속 진행한다. 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu')) #softmax 통과 전까지는 37만 이따위 값도 다나옴. softmax통과해야 
model.add(Dense(10, activation='softmax'))  # 다중분류는 무조건 class의 개수가 마지막 출력 층 개수가 문제다.


#R2기준으로 0.6 이상 만들기
# 컴파일 및 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'], ) # 분류모델은 loss가 실무에서 무조건, binary_crossentropy다. 2개 이상은 리스트이다. 무조건. 더 들어갈 수 있는 공간도 있기에 [ ] 로 묶은것이다. 평가 지표로 accuracy를 넣으면, val_accuracy도 같이 출력을 해준다. 
es = EarlyStopping(monitor='val_accuracy', mode= 'max', patience=100, restore_best_weights=True) #EarlyStopping을 미리 설정을 해 준다. 이거는 콜벡 함수라서 사용할 시 등록을 해주어야 한다.

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=4, validation_split = 0.3, callbacks=[es])
end_time = time.time()

print("======================================================================================================")

# 예측 및 평가
loss = model.evaluate(x_test,y_test) #binary_crossentropy 도 낮으면 좋다. 애초에 모든 loss는 낮으면 좋다. loss, error, cost, 비용, 손실 모두 동의어다. 
print(loss) #[0.2928575575351715, 0.8947368264198303] 가 나온다. 각각 loss, accuracy를 의미한다. 자세히 보면 기존 loss, accuracy를 천장함수 때린 것을 볼 수 있다. 
 
y_predict = np.argmax(model.predict(x_test), axis=1) #predict는 array 혹은 array_like를 요구한다. 그래서 vector넣어도 array 라서 먹힌다.  훈련을 하는 구간이 아니다. (그런데, 실제로는 한다. 그래서 위험하다.--> 특히 중국 모델들.) 순방향만 하기 때문이다. 역방향이 있어야 훈련을 한다. 
# 다중분류에서는 y_predict에 np.round를 씌우면 안된다. 만약 요소가 2개면, 즉 이진 분류에서는 사용해도 된다. 그런데, 요소가 3개 이상이면 np.round를 사용하면 안된다. 
# np.argmax함수를 다중분류에서는 사용한다. 3개 이상의 값 중 가장 큰 값을 1로 만들고 나머지 값은 0으로 바꾼다. 그 후 그 값이 1 0 0 이렇게 나오면 0으로 만들어주는 것 까지 해준다. 
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)
#exit()
#print("y_predict, y.shape", y_predict[:5], y_predict.shape) #predict하면 재학습 하는 부분은 없다. 즉 순방향만 있고 역전파 올라가면서 가중치 갱신하는 부분이 없다는 것이다. 그런데, 실제로 회사들은 여기다 학습을 시키는 코드를 더 껴놓는다. 그래서 개인정보 넣으면 안되는 이유다. 

#r2 = r2_score(y_test, y_predict) r2는 회귀모델에서 사용하며 분류모델에서는 사용하지 않는다. 또한 회귀모델에서 loss는 loss이며, 분류모델에서 loss는 binary_crossentroppy와 같다. 
accuracy = accuracy_score(y_test, y_predict) #애초에, accuracy는 [0, 1, 2, . . . ] 필요하다. 그래서 np.args를 사용하는 것이기도 하다. 
print(accuracy)
print(f"소요 시간: {end_time - start_time}")


# 그냥 하면 ValueError: Classification metrics can't handle a mix of binary and continuous targets 가 나온다. y_predict는 0 에서 1 사이고 y_test하면 0또는 1이기 때문이다. 그래서 무조건 반올림 해야 한다. 
# np.round() 하면 된다. 


# 처음에는 ValueError: Arguments `target` and `output` must have the same rank (ndim). Received: target.shape=(2,), output.shape=(2, 3) 이렇게 나온다. 
# 이번에는 행렬 로 달라고 말을 하는 것이다. 


####### onehot 02 Tensorflow꺼 이용 ############
