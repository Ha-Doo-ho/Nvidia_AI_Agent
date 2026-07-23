# 09_02 카피해서 수정, 성능비교

from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 스코어 
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np 
import time
from xgboost import XGBRegressor

# 09_2를 카피하였다. 

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.95, shuffle=True, random_state=80,)
# 통계의 '층화추출(Stratified Sampling)'에서 유래한 개념입니다. 
# 데이터를 나눌 때(예: 훈련용 데이터와 테스트용 데이터 분리), 전체 데이터가 가진 특정 분류(Class) 비율을 쪼개진 데이터들에서도 동일하게 유지하도록 해주는 핵심 옵션/파라미터를 의미합니다
# 거의 필수로 넣어주는 파라미터이다. 그래서 꼭 np.unique나 value를 세는 메서드를 통해 y의 클래스가 불균형인지 아닌지를 "반드시 " 확인해야 한다. 

# 이제 KFold에도 클래스간 불균형이 있을 수 있다. Kfold 내에서도 test데이터 셋에 특정 클래스가 ㅈㄴ 많거나, 없을 수 있다. 그 생각도 해야 한다. 

# 2. 모델 
parameters = [
    {"n_estimators":[100,200,1100], "max_depth":[6, 10, 12], 'learning_rate':[0.1, 0.09]}, #딕셔너리 사용하였음. 자바에서의 Map이다. 자바에서도 map을 많이 사용하는 것처럼 Python에서도 dictionary가 많이 사용한다.  
    {"max_depth":[5,6,10,12], "learning_rate":[0.1,0.01,0.001, 0.09]},
    {"min_child_weight":[2,3,5, 10], "learning_rate":[0.1, 0.01, 0.001]},
]

kFold = KFold(n_splits=5, shuffle=True, random_state=333)

# 2. 모델
model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kFold, verbose=1,)

# 3. 컴파일, 훈련 
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


# 4 평가 예측
print('model.best_params_: ',model.best_params_)
print('model.score: ', model.score(x_test, y_test))
# 기존: acc 0.875

# model.best_params_:  {'min_child_weight': 2, 'learning_rate': 0.1}
# model.score:  0.8328706299379369 잘 안나온다. 