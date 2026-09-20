import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# 1 데이터

# Autoencoder는 답지(y)가 클래스가 아니다. 답지(y)가 자기 자신이다.
# y_train, y_test를 받아서 사용하지 않는다.
(x_train, _), (x_test, _) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape)

# 1- 2데이터 전처리 (정규화 및 train - val - test로 분리)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float64") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

x_train, x_val = train_test_split(x_train, train_size=0.8, random_state=11, shuffle=True)
print(x_train.shape, x_val.shape)
