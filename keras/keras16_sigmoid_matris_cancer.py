import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

#keras14_EarlyStopping2_diabets.py 를 카피하였다. 
#이진분류에서 사용하는 지표는 단 1개 뿐이다. binary_crossentropy 뿐이다.


#1 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR) #조금 더 쉽게 알아볼 수 있다. 그래서 보통 이걸 사용한다.
#print(datasets.feature_names) 빠르게 열에 어떤 데이터 특성이 있는지 알 수 있다. 
print(datasets.feature_names) # 열 이름이다. feature, attrubute, 특징, 열 모두 같은 말이다. 
#exit()

#x,y 자체가 섞여있다. 데이터 셋을 분리해야 한다. sklearn은 x,y를 섞어놨다. 그래서 데이터 자체를 분리해야 한다. 
# 실무에서는 .data, .target 안쓰고 pandas로 read_csv, to_csv를 사용해야 한다.  
x = datasets.data 
y = datasets.target
print("x: ",x)
print("y: ",y)

print(x.shape, y.shape) #(569, 30) (569,)
print(np.unique(y, return_counts=True)) #(array([0, 1]), array([212, 357])) --> 이건 0은 212개, 1이357개 있다는 의미이다. 
#exit()

# train, test로 자른다. 
x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=30, shuffle=True)
print(x_train.shape, x_test.shape) #(455, 30) (114, 30)
print(y_train.shape, y_test.shape) #(455,) (114,)
#exit()

# 모델 구성
model = Sequential()
model.add(Dense(10,input_shape=(30,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(68, activation='relu'))
model.add(Dense(126, activation='relu'))
model.add(Dense(258, activation='relu'))
model.add(Dense(516, activation='relu'))
model.add(Dense(258, activation='relu'))
model.add(Dense(126, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  #디폴트가 linear


#R2기준으로 0.6 이상 만들기
# 컴파일 및 훈련
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'], ) # 분류모델은 loss가 실무에서 무조건, binary_crossentropy다. 2개 이상은 리스트이다. 무조건. 더 들어갈 수 있는 공간도 있기에 [ ] 로 묶은것이다. 평가 지표로 accuracy를 넣으면, val_accuracy도 같이 출력을 해준다. 
es = EarlyStopping(monitor='val_loss', mode= 'min', patience=100, restore_best_weights=True) #EarlyStopping을 미리 설정을 해 준다. 이거는 콜벡 함수라서 사용할 시 등록을 해주어야 한다.

model.fit(x_train, y_train, epochs=100000, batch_size=2, validation_split = 0.3, callbacks=[es])

print("======================================================================================================")
# 예측 및 평가
loss = model.evaluate(x_test,y_test) #binary_crossentropy 도 낮으면 좋다. 애초에 모든 loss는 낮으면 좋다. loss, error, cost, 비용, 손실 모두 동의어다. 
print(loss) #[0.2928575575351715, 0.8947368264198303] 가 나온다. 각각 loss, accuracy를 의미한다. 자세히 보면 기존 loss, accuracy를 천장함수 때린 것을 볼 수 있다. 
#exit()     #그리고 여기서도 val_accuracy, val_loss를 믿으면 된다. 
y_predict = model.predict(x_test) #predict는 array 혹은 array_like를 요구한다. 그래서 vector넣어도 array 라서 먹힌다.  훈련을 하는 구간이 아니다. (그런데, 실제로는 한다. 그래서 위험하다.--> 특히 중국 모델들.) 순방향만 하기 때문이다. 역방향이 있어야 훈련을 한다. 
y_predict = np.round(y_predict) # 추가한 것.  sigmoid를 적용하면 0.0 초과 1.0 미만의 값으로 나온다. 그런데, y값인 답지를 출력해보면 죄다 0아니면, 1이다. 그래서 round로 반올림을 해야 한다. 그래서 np.round()가 필요하다. 
print("y_predict", y_predict) #predict하면 재학습 하는 부분은 없다. 즉 순방향만 있고 역전파 올라가면서 가중치 갱신하는 부분이 없다는 것이다. 그런데, 실제로 회사들은 여기다 학습을 시키는 코드를 더 껴놓는다. 그래서 개인정보 넣으면 안되는 이유다. 

#r2 = r2_score(y_test, y_predict) r2는 회귀모델에서 사용하며 분류모델에서는 사용하지 않는다. 또한 회귀모델에서 loss는 loss이며, 분류모델에서 loss는 binary_crossentroppy와 같다. 
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

# 그냥 하면 ValueError: Classification metrics can't handle a mix of binary and continuous targets 가 나온다. y_predict는 0 에서 1 사이고 y_test하면 0또는 1이기 때문이다. 그래서 무조건 반올림 해야 한다. 
# np.round() 하면 된다. 


