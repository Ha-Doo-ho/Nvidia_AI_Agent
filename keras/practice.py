import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd
import time 

# 1 데이터
path = "./_data/" #현재는 Terminal보면  C:\nsu_work> 라고 있는 것을 볼 수 있다. 이것이 현재 디렉터리이다. 현재는 . 과거는 .. 이건 Linux랑 같다.
train_csv = pd.read_csv(path+"train.csv")
test_csv = pd.read_csv(path+"test.csv")
submit_csv = pd.read_csv(path + "sampleSubmission.csv")

print(test_csv) # 이거 한번 해보면 가장 왼쪽에 0 1 2 3 4 5 6 . . . 이렇게 붙어 있는 것을 볼 수 있다. 이건 vscode 같은 곳에서 보기 편하라고 만든 것이고 실제로는 의미 없다. 
                # 그리고 그것들을 index라고 부르는데, 데이터로 취급하지 않는다. 우리는 datetime을 인덱스로 취급할 것이다. 즉, 데이터로 취급하지 않겠다는 것이다.
                # 그것을 만들어 주는 것이 index_col = 0 0은 datetime이니까 그걸 인덱스로 만든 것이다. 이다. 이것을 추가해서 보게 되면, datetime이 기존엔 column에 있었는데, index위치로 간 것을 볼 수 있다. 
                # 그 파일 3개에 모두 똑같이 적용해야 하니까 index_col=0을 전부 붙인다. 
x = train_csv.drop(['','',''])

# 2 모델 구성

# 3 컴파일 및 훈련

# 4 예측 및 평가 