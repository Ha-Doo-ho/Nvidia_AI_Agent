# PCA는 딱 한가지 용도로 사용한다. 컬럼축소용 으로만 사용한다. 
# 11_1에서 #1 데이터 부분만 땡기자

from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 스코어 
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np 
import time
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
# PCA를 배워보자


# 08_1을 카피하였다.                                   
# 이제, 파라미터 튜닝을 자동화 해보자 

x, y = load_breast_cancer(return_X_y=True)

pca = PCA(n_components=10) #n_components: 해당 개수로 압축한다. 기존에 30개의 컬럼을 가졌지만 10개로 압축한다. 컬럼이 많으면, 이걸 사용하면 된다.
x = pca.fit_transform(x) #이것도 전처리다. 그래서 전처리 실행 명령어는 전부 fit_transform()으로 되어 있다. 
print(x.shape)
exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=333,stratify=y)

print(x_train.shape, x_test.shape) #(455, 30) (114, 30)
print(y_train.shape, y_test.shape) #(455,) (114,)

print(y_test) #2진분류


#10_2 #2부터 끝까지 카피
# 2. 모델

parameters = {
    "learning_rate":0.1,
    "max_depth":6,
    "n_estimators":200,
}

model = XGBClassifier(**parameters)
# ** 2개는 딕셔너리 불러오라는 것이다. * 1개는 리스트 불러오라는 것이다. 즉 parameters라는 딕셔너리를 불러오라는 뜻이다. 이전보다 더욱 깔끔해졌다. 

# 3. 컴파일, 훈련 
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 저게 5등분이 아니어도 된다. 10등분이어도 된다. 대신 10번을 도는 것이다. 

# KFold가 완벽한줄 알았다. 사실 데이터가 적으면 상당히 좋다. 최대한 많이 훈련할 수 있기 때문이다. 

# 분류에서는 클래스간 불균형. 즉 왜도가 생겼을 때에 정확한 분류를 못할 수도 있다. 정확한 분류를 하려면 클래스 간의 균형이 무조건 필요하다.  
# 이 66개의 경우의 수(candidates) 중 가장 좋은 파라미터만 잡는다면 그 이후로는 GridSearchCV를 할 필요가 없을 것이다. 그것을 만든 사람들도 알고 있을 것

# 최적의 매개변수를 찾아보자 
#print("최적의 매개변수:", model.best_estimator_)
#최적의 매개변수: SVC(C=1, kernel='linear')

#print("최적의 파라미터(이름 같음): ", model.best_params_) #최적의 파라미터(이름 같음):  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
#최적의 파라미터(이름 같음):  {'C': 1, 'degree': 3, 'kernel': 'linear'} 조금 더 정확하게 볼 수 있다. 더 익숙하기도 하다. --> 중요하다. 이거 가지고 어떤 파라미터 사용했을 때 제일 잘 나왔는지 알 수 있다. 

# 4 평가 예측
print('model.score: ', model.score(x_test, y_test))  # 이건 정말 실전인 test데이터를 가지고 만든 스코어이다. 그래서 bestscore를 믿지말고, score을 믿어야 한다. 
#model.score:  0.8291540612455008 이거 회귀니까 r2로 자동으로 잡아준다. 그래서 r2값과 똑같이 나온다. 
#model.score:  0.8371457284900378
#model.score:  0.8394807521086485
y_predict = model.predict(x_test)
print(f"걸린시간{np.round(end_time-start_time, 4)} 초")


#여기서 알아가야 할 것은 .best_params_ 와 .model.best_score_ 이 된다. 