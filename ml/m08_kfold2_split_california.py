from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 스코어 
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np 

# 07_1을 카피하였다.                                   

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_tset = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=11,) #startify=true는 분류 모델이다. 
# 통계의 '층화추출(Stratified Sampling)'에서 유래한 개념입니다. 
# 데이터를 나눌 때(예: 훈련용 데이터와 테스트용 데이터 분리), 전체 데이터가 가진 특정 분류(Class) 비율을 쪼개진 데이터들에서도 동일하게 유지하도록 해주는 핵심 옵션/파라미터를 의미합니다
# 거의 필수로 넣어주는 파라미터이다. 그래서 꼭 np.unique나 value를 세는 메서드를 통해 y의 클래스가 불균형인지 아닌지를 "반드시 " 확인해야 한다. 

# 이제 KFold에도 클래스간 불균형이 있을 수 있다. Kfold 내에서도 test데이터 셋에 특정 클래스가 ㅈㄴ 많거나, 없을 수 있다. 그 생각도 해야 한다. 

print(y_train)
print(y_tset)
#exit()

KFold = KFold(n_splits=5, shuffle=True) # 5등분을 하더라도 섞어야 한다. 디폴트 값이 false라서 suffle=True가 train_test_split과 다르게 반드시 요구된다. 
#KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123) # 이건 분류에서 사용하는 것이다. 왜? KFold에서도 클래스 간 불균형을 막기 위해서 사용하는 것이니까, 당연히 회귀에서는 기본 KFold를 사용하는 것이다.  
# 2. 모델
#model = DecisionTreeClassifier()
#model = DecisionTreeRegressor()
model = RandomForestRegressor( ) # n_estimators: epoches와 비슷한 개념이다. 문제는, 이런 파라미터 튜닝하다는 것이 너무 힘들다는 것이다. 

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

# 저게 5등분이 아니어도 된다. 10등분이어도 된다. 대신 10번을 도는 것이다. 

# KFold가 완벽한줄 알았다. 사실 데이터가 적으면 상당히 좋다. 최대한 많이 훈련할 수 있기 때문이다. 

# 분류에서는 클래스간 불균형. 즉 왜도가 생겼을 때에 정확한 분류를 못할 수도 있다. 정확한 분류를 하려면 클래스 간의 균형이 무조건 필요하다.  
