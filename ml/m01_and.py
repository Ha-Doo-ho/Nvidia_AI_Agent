import numpy as np 
from sklearn.svm import LinearSVC #서포트 벡터 머신이다. 선형 회귀 모델로 알다시피 ㅈㄴ 오래된 모델이라서 이런 것이 있다고만 생각할 것
from sklearn.metrics  import accuracy_score

# 1. 데이터
x_data = np.array([[0,0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 0, 0, 1])
print(x_data.shape, y_data.shape)#(4, 2) (4,)


# 2. 모델 
# 머신러닝은 대부분 sklenarn에서 제공을 한다. 
# 모델은 무거워서 여러개 필요한데, 머신러닝 같은 경우는 알고리즘에 가까워서, 그냥 가져다 쓰면 된다. 
model = LinearSVC()

# 3. 훈련 --> loss자체가 다 있어서 괜찮음
model.fit(x_data,y_data)

# 평가 예측
y_predict = model.predict(x_data)

# 모든 머신러닝 모델들도 전부 fit, predict 이 똑같다. 그래서 외울 필요가 없다. 

results = model.score(x_data, y_data) #sklearn은 score이다. tensorflow의 evaluate랑 같다.  이진분류니까 평가지표가 accuracy로 잡는다. 
                                      #회귀모델도 score로 쓰는 것은 똑같다. 그런데, 내부는 회귀니까 평가지표를 r2_score로 해준다.  
print(results)

acc = accuracy_score(y_data, y_predict)
print("acc : ", acc)