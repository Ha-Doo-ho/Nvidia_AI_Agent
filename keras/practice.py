import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd
import time 
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine

# 1 데이터
datasets = load_wine()
#print(datasets.DESCR) # 클래스 개수 3개
x = datasets.data
y = datasets.target 
print(x.shape)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=11, shuffle=True)
print(x_train.shape)
exit()

# 2 모델 구성
model = Sequential()
model.add(2, Dense(input_shape=(13, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu')) #이 개수를 바꿔보는 것이 하이퍼 파라미터 튜닝 (층의 개수를 바꿔보는 것이다.)
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류는 무조건 class의 개수가 마지막 출력 층 개수가 문제다.


# 3 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', patience=100,restore_best_weights=True, mode='max')
start_time = time.time()
model.fit(x_train, y_train, batch_size=32, epochs=3, callbacks=[es], validation_split=0.7, verbose=1)
end_time = time.time()

# 4 예측 및 평가 
loss = model.evaluate(x_test, y_test)

np.argmax(model.predict(x_test))