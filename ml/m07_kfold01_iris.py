from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 스코어 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np 

x, y = load_iris(return_X_y=True)

KFold = KFold(n_splits=5, shuffle=True) # 5등분을 하더라도 섞어야 한다. 디폴트 값이 false라서 suffle=True가 train_test_split과 다르게 반드시 요구된다. 

# 2. 모델
model = DecisionTreeClassifier()

# 3. 컴파일, 훈련 

 
# 4. 평가 예측 
scores = cross_val_score(model, x,y, cv=KFold, n_jobs=-1) #-1로 하면 cpu로 돌아감. 머신러닝은 cpu로 돌아감 
print('ACC : ', scores, '\n cross_val_score 평균: ', round(np.mean(scores), 4))

#  [0.96666667 1.         0.96666667 0.93333333 0.96666667] 교차검증점수에 의해 나온 값 
# 첫번째 결과 0.96666667
# 두번째 결과 1.
# 세번째 결과 0.96666667
# 네번째 결과 0.93333333
# 다섯번째 결과 0.96666667