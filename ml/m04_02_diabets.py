import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_diabetes

# 데이터
x, y = load_diabetes(return_X_y=True) #이런 방식으로도 불러올 수 있다. 예전의 .data, .target으로도 상관 없음
print(x.shape, y.shape) #(442, 10) (442,)

#exit()
# 2. 모델 구성
# 텐서플로우로 안만들고 legacy한 레거시 모델을 만들자
from sklearn.svm import LinearSVC #서포트 벡터 머신
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor  # 수치(0.xxxx 같이 소수점 나오는 것)면 회귀모델이며, 0,1,2 같이 숫자 딱딱 떨어지면 분류
from sklearn.ensemble import RandomForestRegressor #나무를 보았으니 숲을 보자. 회귀를 하고 있으니까 Regressor 이다. 나무가 모였으니 앙상블이라고 하고, DecisionTreeRegressor을 앙상블해서, 랜덤포레스트가 된 것이다.  
#Tree구조는 매우 강력해서 성능도 매우 좋다. tensorflow보다 더 뛰어나다.

#model = LinearSVC() # 서포트벡터머신 실행. 이건 분류임. 성능도 구림. ㅈㄴ 옛날에 만들어진 알고리즘에 가까움.  r2: 0.14027149321266968
#model = LogisticRegression() #로지스틱 회귀이다. 놀랍게도 이름에 회귀가 들어갔음에도 불구하고 "분류이다." 실무에서는 안씀. 너무 구림. r2: 0.020361990950226245 
#model = DecisionTreeRegressor() # 머신러닝은 단층 퍼셉트론이라서 매우 빠르다. 성능도 오히려 tensorflow, pytorch 보다 좋은 것이 많다. 머신러닝이지만 kaggle에서 사용되는 모델이 다 DecisionTreeRegressor 기반이다.  r2: 1.0
model = RandomForestRegressor() # r2: 0.9199918300378853. 튜닝을 안하면 의사결정트리보다 성능이 안좋다. 만약 train_test_split이다. validation 등 여러개를 걸어주면 의사결정트리보다 훨씬 좋다.
# 그래서 이 랜덤포레스트를 개량한 것이 xgboast, LGBM, Catboast 이며, 실무와 대회에서 모두 사용한다. 

# 3. 컴파일, 훈련
model.fit(x,y)

# 4. 평가 예측
results = model.score(x,y) #회귀모델은 r2로 걸리고, 분류모델은 accuracy로 걸린다. 자동으로 해줌 여기서는 분류니까 알아서 accuracy로 걸어줌. 
print(results)

#shape도 맞추지 않았다. 