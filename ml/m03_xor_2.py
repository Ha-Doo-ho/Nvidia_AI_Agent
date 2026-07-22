import numpy as np 
from sklearn.svm import LinearSVC #서포트 벡터 머신이다. 선형 회귀 모델로 알다시피 ㅈㄴ 오래된 모델이라서 이런 것이 있다고만 생각할 것
from sklearn.metrics  import accuracy_score
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense

# 단층 퍼셉트론을 텐서플로우로 구현한 것 뿐이다. 

# 1. 데이터
x_data = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])
print(x_data.shape, y_data.shape)#(4, 2) (4,)

# 2. 모델 
# 머신러닝은 대부분 sklenarn에서 제공을 한다. 
# 모델은 무거워서 여러개 필요한데, 머신러닝 같은 경우는 알고리즘에 가까워서, 그냥 가져다 쓰면 된다. 
#model = LinearSVC()
#model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) # 히든 레이어가 없는 

# 3. 컴파일 및 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #acc라고 써도 됨. 
model.fit(x_data,y_data, batch_size=1, epochs=100)

# 평가 예측
results = model.evaluate(x_data, y_data)
print("results", results)
y_predict = np.round(model.predict(x_data))

# 모든 머신러닝 모델들도 전부 fit, predict 이 똑같다. 그래서 외울 필요가 없다. 

#results = model.score(x_data, y_data) #sklearn은 score이다. tensorflow의 evaluate랑 같다.  이진분류니까 평가지표가 accuracy로 잡는다. 
                                      #회귀모델도 score로 쓰는 것은 똑같다. 그런데, 내부는 회귀니까 평가지표를 r2_score로 해준다.  
print(results)

acc = accuracy_score(y_data, y_predict)
print("acc : ", acc)
