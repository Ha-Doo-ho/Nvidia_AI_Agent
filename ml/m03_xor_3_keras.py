# m_02 카피하였음 
# 다층퍼셉트론 구성으로 xor 인공지능 겨울 문제 해결 
import numpy as np 
from sklearn.svm import LinearSVC #서포트 벡터 머신이다. 선형 회귀 모델로 알다시피 ㅈㄴ 오래된 모델이라서 이런 것이 있다고만 생각할 것
from sklearn.metrics  import accuracy_score
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 1])
print(x_data.shape, y_data.shape)#(4, 2) (4,)


# 2. 모델 
# 머신러닝은 대부분 sklenarn에서 제공을 한다. 
# 모델은 무거워서 여러개 필요한데, 머신러닝 같은 경우는 알고리즘에 가까워서, 그냥 가져다 쓰면 된다. 
#model = LinearSVC()
model = Sequential()
model.add(Dense(20, input_shape=(2,), activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 훈련 --> loss자체가 다 있어서 괜찮음
model.compile(loss='binary_crossentropy' ,optimizer='adam',)
model.fit(x_data,y_data, epochs=100,batch_size=1)



# 평가 예측
y_predict = np.round(model.predict(x_data))

# 모든 머신러닝 모델들도 전부 fit, predict 이 똑같다. 그래서 외울 필요가 없다. 


acc = accuracy_score(y_data, y_predict)
print("acc : ", acc)