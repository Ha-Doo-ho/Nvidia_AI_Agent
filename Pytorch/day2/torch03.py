import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

import torch
from torch.cuda import is_available
from torch import tensor
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, ReLU, Dropout, Sigmoid, BCELoss
from torch.optim import Adam

# 0 디바이스 설정
if is_available():
    device = "cuda"
else:
    device = "cpu"

# 1 데이터 --> 텐서로 만들고 train데이터 셋은 DataLoader로 추가 변환
def makeDataset(x_train, x_val, y_train, y_val, batch_size=32):
    x_train_tensor = tensor(x_train)
    x_val_tensor = tensor(x_val)
    y_train_tensor = tensor(y_train)
    y_val_tensor = tensor(y_val)
    
    train_Dataset = TensorDataset(x_train_tensor, y_train_tensor) # 문제 - 정답 한쌍(Pair)으로 묶이게 된다.
    train_DataLoader = DataLoader(dataset=train_Dataset, batch_size=batch_size)
    
    return train_DataLoader, x_val_tensor, y_val_tensor

# 3 컴파일 및 훈련
# device, 데이터 로더, 모델, loss_fn, optimizer가 필요함
# 핵심은 미니배치 단위로 학습을 시켜야 하므로, 미니 배치 단위의 학습을 하는 함수로 만들어야 함.
def train(device, model, train_DataLoader, loss_fn, optimizer):
    num_batches = len(train_DataLoader)
    train_loss = 0
    
    model.train() # 학습 모드로 전환
    
    for batch_idx, (x,y) in enumerate(train_DataLoader):
        x = x.to(device)
        y = y.to(device)
        
        # 순전파 feed-Forword
        pred = model(x)
        loss = loss_fn(pred, y)
        train_loss += loss
        
        # 역전파 Backpropagation
        loss.backward() # loss를 이용해서 기울기를 계산하여 구함
        optimizer.step() # 구한 기울기로 가중치를 수정함
        optimizer.zero_grad() # 기울기 초기화. 없으면 기울기가 계속 쌓인다. loss.backward()가 .zer_grad가 없으면 기울기를 계속 누적하게 만들었다.
    
    train_loss /= num_batches
    return train_loss.item()
     
# 4 평가   
def evaluate(device, model, x_val_tensor, y_val_tensor, loss_fn):
    model.eval() #평가 모드로 전환
    
    with torch.no_grad(): #기울기는 더이상 계산할 필요가 없으므로 기울기 계산 기능을 종류함
        x = x_val_tensor.to(device)
        y = y_val_tensor.to(device)
        
        pred = model(x)
        eval_loss = loss_fn(pred, y).item()
        
        return pred, eval_loss
        
# 1 데이터
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/titanic.3.csv"
data = pd.read_csv(path)
print(data.head())
print(data.shape) #(891, 11)

# 1-1 데이터 분할
x = data.loc[: , ["Pclass", "Sex", "Age"]]
y = data.loc[:, "Survived"]
print(x.shape, y.shape) # (891, 11)
print(type(x), type(y)) # <class 'pandas.DataFrame'> <class 'pandas.Series'>

# 1-2가변수화
x = pd.get_dummies(data=x, columns=["Sex"], drop_first=True)
print(x.head())

# 1-3 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x,y, train_size=0.8, random_state=11, stratify=y)

# 1-4 스케일링
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

print("x_train & x_val type",type(x_train))  #x_train & x_val type <class 'numpy.ndarray'>

# 2 모델구성
