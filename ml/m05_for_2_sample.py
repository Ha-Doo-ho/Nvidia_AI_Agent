# enumerate를 이용해서 모델을 전부 돌려보자. 
#04_2 카피 
import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# 데이터
data_list = [
    load_wine(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True)
]

model_list = [
    LinearSVC(),
    LogisticRegression(), #말은 회귀인데, 신기하게도 분류임. 그래서 면접이나 빅데이터 분석기사에 함정문제로 잘 나온다. 
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

data_name_list = ['아이리스: ', '브래스트캔서: ', '와인: ']
model_name_list = ['LinearSVC: ', "LogisticRegression : ", 'DecisionTree: ', 'RandomForest: '] #RandomForest는 RF 라고도 사용한다. 
#exit()

# 2. 모델 구성
for i, value in enumerate(data_list): #enumerate is useful for obtaining an indexed list: enumerate에는 통상적으로 리스트가 들어간다. 
    x, y = value
    print("======================================")
    print(data_name_list[i])
    print(x.shape, y.shape)
    
    for j, value2 in enumerate(model_list):
        model = value2
        
        #3. 컴파일 및 훈련
        model.fit(x,y)
        
        #4 평가, 예측 
        results = model.score(x,y)
        print(model_name_list[j], "model.score: ", results)
exit()
#model = LinearSVC() # 서포트벡터머신 실행. 이건 분류임. 성능도 구림. ㅈㄴ 옛날에 만들어진 알고리즘에 가까움.  r2: 0.14027149321266968
#model = LogisticRegression() #로지스틱 회귀이다. 놀랍게도 이름에 회귀가 들어갔음에도 불구하고 "분류이다." 실무에서는 안씀. 너무 구림. r2: 0.020361990950226245 
#model = DecisionTreeRegressor() # 머신러닝은 단층 퍼셉트론이라서 매우 빠르다. 성능도 오히려 tensorflow, pytorch 보다 좋은 것이 많다. 머신러닝이지만 kaggle에서 사용되는 모델이 다 DecisionTreeRegressor 기반이다.  r2: 1.0
#model = RandomForestRegressor() # r2: 0.9199918300378853. <튜닝을 안하면 의사결정트리보다 성능이 안좋다.> 만약 train_test_split이다. validation 등 여러개를 걸어주면 의사결정트리보다 훨씬 좋다.
# 그래서 이 랜덤포레스트를 개량한 것이 xgboast, LGBM, Catboast 이며, 실무와 대회에서 모두 사용한다. 
