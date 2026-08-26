import numpy as np
import tensorflow as tf 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler #이상치와 결측치 처리에서 좋음
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor # 나무를 합친 것(앙상블) 이니까 숲이다. 
from sklearn.utils import all_estimators

# 1 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=11, train_size=0.7, shuffle=True)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) #기존에는 fit하고 transform 했지만 이제 더 줄어들었다. 
x_test = scaler.transform(x_test)

# 2 모델 구성 
# model = RandomForestRegressor()
all_models = all_estimators(type_filter='regressor') # 사이킷 런에서 제공하는 회귀모델의 개수이다. 
print("all_models : ", all_models)
print("모델의 개수: ", len(all_models)) # len() 객체의 수를 출력한다. 

max_score = 0
max_name = '바보'
for (name, algorithm) in all_models:
    #print(v) #enumarater가 아니니까 인덱스: 값이 아닌 바로 값이 들어간다. 
    try: #에러뜨면 바로 except로 넘어가고, 그렇지 않으면try로 간다. 
        model = algorithm()
        
        # 3. 훈련
        model.fit(x_train, y_train)
        
        # 4. 평가 예측 
        results = model.score(x_test, y_test)
        print(name, '의 정답률 ', results)
        
        if(max_score < results):
            max_score = results
            max_name = name 
    except:
        print(name, '은(는) 에러뜬 분.')

print("=============================================")
print(f"max_score:{max_score},  max_name:{max_name}")
print("=============================================")

