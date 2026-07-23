from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 스코어 
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np 
import time
from xgboost import XGBRegressor

from sklearn.svm import SVC, LinearSVC

# 09_1을 카피하였다.                                   
# 이제, 파라미터 튜닝을 자동화 해보자 

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.95, shuffle=True, random_state=80,)
# 통계의 '층화추출(Stratified Sampling)'에서 유래한 개념입니다. 
# 데이터를 나눌 때(예: 훈련용 데이터와 테스트용 데이터 분리), 전체 데이터가 가진 특정 분류(Class) 비율을 쪼개진 데이터들에서도 동일하게 유지하도록 해주는 핵심 옵션/파라미터를 의미합니다
# 거의 필수로 넣어주는 파라미터이다. 그래서 꼭 np.unique나 value를 세는 메서드를 통해 y의 클래스가 불균형인지 아닌지를 "반드시 " 확인해야 한다. 

# 이제 KFold에도 클래스간 불균형이 있을 수 있다. Kfold 내에서도 test데이터 셋에 특정 클래스가 ㅈㄴ 많거나, 없을 수 있다. 그 생각도 해야 한다. 

print(y_train)
print(y_test)
#exit()

# 2. 모델 
parameters = [
    {"n_estimators":[100,200,1100], "max_depth":[6, 10, 12], 'learning_rate':[0.1, 0.09]}, #딕셔너리 사용하였음. 자바에서의 Map이다. 자바에서도 map을 많이 사용하는 것처럼 Python에서도 dictionary가 많이 사용한다.  
    {"max_depth":[5,6,10,12], "learning_rate":[0.1,0.01,0.001, 0.09]},
    {"min_child_weight":[2,3,5, 10], "learning_rate":[0.1, 0.01, 0.001]},
]  # SVC는 어차피 안써서 필요가 없다. 그러나 XGboost는 파라미터를 알아야 한다. 트리계열의 부스팅 계열 모델이다. 
    # 18 + 12 + 12 = 42번



kFold = KFold(n_splits=5, shuffle=True, random_state=333) # 이건 분류에서 사용하는 것이다. 왜? KFold에서도 "클래스 간 불균형"을 막기 위해서 사용하는 것이니까, 당연히 회귀에서는 기본 KFold를 사용하는 것이다.  

# 2. 모델
# 여기에서 parameters를 사용한다. 그러려면 import가 필요하다. 
#model = SVC()  
model = GridSearchCV(XGBRegressor(), parameters, cv=kFold, verbose=1,) # SVC()라는 모델을 감쌀것이다. param_grid: Mapping | Sequence[dict] 라서 parameters가 딕셔너리가 가능하다. 
                                                              # dict or list of dictionaries 를 요구한다. 
                                                              # GridSearchCV 자체는 모델이 아니고, warpping하기 위해 존재한다. 
                                                              # Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 3. 컴파일, 훈련 
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 저게 5등분이 아니어도 된다. 10등분이어도 된다. 대신 10번을 도는 것이다. 

# KFold가 완벽한줄 알았다. 사실 데이터가 적으면 상당히 좋다. 최대한 많이 훈련할 수 있기 때문이다. 

# 분류에서는 클래스간 불균형. 즉 왜도가 생겼을 때에 정확한 분류를 못할 수도 있다. 정확한 분류를 하려면 클래스 간의 균형이 무조건 필요하다.  
# 이 66개의 경우의 수(candidates) 중 가장 좋은 파라미터만 잡는다면 그 이후로는 GridSearchCV를 할 필요가 없을 것이다. 그것을 만든 사람들도 알고 있을 것

# 최적의 매개변수를 찾아보자 
print("최적의 매개변수:", model.best_estimator_)
#최적의 매개변수: SVC(C=1, kernel='linear')

print("최적의 파라미터(이름 같음): ", model.best_params_) #최적의 파라미터(이름 같음):  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
#최적의 파라미터(이름 같음):  {'C': 1, 'degree': 3, 'kernel': 'linear'} 조금 더 정확하게 볼 수 있다. 더 익숙하기도 하다. --> 중요하다. 이거 가지고 어떤 파라미터 사용했을 때 제일 잘 나왔는지 알 수 있다. 

# 4 평가 예측
print('best_score: ', model.best_score_)
# best_score:  0.8414536940663545 --> train 데이터에 되어 있는, 과적합되어 있는 스코어이다. 그래서 높게나올 수 밖에 없는 것이다.  

# 즉 best_score:  {'C': 1, 'degree': 3, 'kernel': 'linear'} 일때,  0.9859605911330049 점수가 나온다는 것이다. 
# 최적의 파라미터(이름 같음):  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 900}
print('model.score: ', model.score(x_test, y_test))  # 이건 정말 실전인 test데이터를 가지고 만든 스코어이다. 그래서 bestscore를 믿지말고, score을 믿어야 한다. 
#model.score:  0.8291540612455008 이거 회귀니까 r2로 자동으로 잡아준다. 그래서 r2값과 똑같이 나온다. 
#model.score:  0.8371457284900378
#model.score:  0.8394807521086485
y_predict = model.predict(x_test)
print('r2', r2_score(y_test, y_predict)) # r2 0.8516948379364527
print(f"걸린시간{np.round(end_time-start_time, 4)} 초")


#여기서 알아가야 할 것은 .best_params_ 와 .model.best_score_ 이 된다. 