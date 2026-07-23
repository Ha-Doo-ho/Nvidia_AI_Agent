from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 스코어 
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np 


# 7_1을 카피하였다,. 

x, y = fetch_california_housing(return_X_y=True)

kFold = KFold(n_splits=5, random_state=11, shuffle=True) # 5등분을 하더라도 섞어야 한다. 디폴트 값이 false라서 suffle=True가 train_test_split과 다르게 반드시 요구된다. 

# 2. 모델
model = DecisionTreeRegressor() #xgboost, lgbm, catboost 도 각각 Classifier, Regressor 2가지 버전으로 나누어져 있다. 
model1 = RandomForestRegressor(max_depth=5, max_leaf_nodes=3, n_estimators=100, max_features=33)
# 3. 컴파일, 훈련 

 
# 4. 평가 예측 
scores = cross_val_score(model, x,y, cv=kFold, n_jobs=-1) # cpu로 돌아감. 머신러닝은 cpu로 돌아감. -1은 통상 전체를 의미한다. cpu개수를 선택할 수 있는데, -1은 cpu전체를 쓰는 것이다. 잘못쓰면 터진다. 
scores1 = cross_val_score(model1, x,y, cv=kFold, n_jobs=-1) # 여기서 n_estimators는 파라미턱다 있다.  model.fit()의 epochs 와 같은 개념이다. 훈련 회수를 의미한다. 

print('ACC : ', scores, '\n cross_val_score 평균: ', round(np.mean(scores), 4))
print('ACC : ', scores1, '\n cross_val_score 평균: ', round(np.mean(scores1), 4))

#  [0.96666667 1.         0.96666667 0.93333333 0.96666667] 교차검증점수에 의해 나온 값 
# 첫번째 결과 0.96666667
# 두번째 결과 1.
# 세번째 결과 0.96666667
# 네번째 결과 0.93333333
# 다섯번째 결과 0.96666667

# 저게 5등분이 아니어도 된다. 10등분이어도 된다. 대신 10번을 도는 것이다. 

#전용 메모리, 공유 메모리가 존재한다. 
# 시스템 메모리(RAM카드) 
# GPU(12GB)

# 머신러닝 모델들은 대부분 CPU를 사용한다. 그게 n-jobs라는 파라미터에 존재하며 그게 cpu를 몇개 쓸것이냐 라는 뜻이다. 그런데, xgboost, lgbm, catboost는 GPU를 사용한다. 
# 그런데, 이것 역시 과적합의 위험이 있다. AI에서 문제는 반 이상이 과적합이 있다. 
# 왜? 아무리 Train데이터로 훈련시키고, Test로 검증한다고 해도, 5번을 나누어서 훈련시킨다고 해도, 결국 모든 데이터를 훈련시킨다. 
# 문제는 내가 훈련시킨 데이터 '외' 다른 데이터를 넣으면, 그것을 잘 맞출 수 있을까? 그러면 다시 과적합 문제로 돌아오는 것이다.  
# 결국 방법은 훈련에 사용하지 않은 새로운 데이터를 사용하는 것이다. 만약 딱 1번만 사용한다고 하면 상관 없다. 그런데, 파라미터 튜닝을 하다 보면 과적합이 된다. 
# 그러면 다시 훈련에 사용하지 않을 데이터를 또 분리를 해야 한다. --> 그래서 train_val_test로 다시 돌아오는 것이다. 
# 우리가 필요한 것은 훈련에 사용되지 않는 데이터 셋이다. 그래서 KFold를 train|val|test 에서 train에 KKold를 적용시키는 것이다. 