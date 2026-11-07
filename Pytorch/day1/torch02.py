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


# 1 데이터 --> Data를 텐서로 만들되,  학습에 사용할 것들은 DataLoader로 넣어야 함. 검증/실제 테스트는 안넣음. 마치 시험은 한번에 치지, 문제 n개씩 주고 n개씩 풀라고 하지 않듯이. 
def makeDataSet(x_train, x_val, y_train, y_val, batch_size=32):
    x_train_tensor = tensor(data=x_train, dtype=torch.float)
    x_val_tensor = tensor(data=x_val, dtype=torch.float)
    
    y_train_tensor = tensor(data=y_train.to_numpy(), dtype=torch.float).reshape(-1,1)
    y_val_tensor = tensor(data=y_val.to_numpy(), dtype=torch.float).reshape(-1,1)
    
    train_Dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_DataLoader = DataLoader(dataset=train_Dataset, batch_size=batch_size)
    
    return train_DataLoader, x_val_tensor, y_val_tensor
    
# 3 컴파일 및 훈련
# 데이터로더 -> train_DataLoader, loss_fn, optimizer, model, device(cpu에 넣을 것이냐 gpu에 넣을 것이냐)
def train(model, device, train_DataLoader, loss_fn, optimizer):
    num_batches = len(train_DataLoader) # batch_size 수가 DataLoader의 수와 같음
    train_loss = 0
    
    model.train()
    
    for batches_idx, (x,y) in enumerate(train_DataLoader): #train_DataLoader는 문제-정답 한쌍(Pair)으로 구성되어 있다.
        x = x.to(device) # 데이터를 학습하기 전에 CPU에 넣을 건지, GPU에 넣을건지 정할 수 있음. 
        y = y.to(device)
        
        # 순전파 feedForward
        pred = model(x)  
        loss = loss_fn(pred, y) 
        train_loss += loss 
        
        # 역전파 Backpropagation 
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
        
            
    train_loss /= num_batches 
    
    return train_loss.item()
        
# 4 평가 및 예측 --> model, device, val_tensor, loss_fn  --> '측정' 하는 단계이므로 가중치를 변경하면 안됨. 그래서 가중치를 변경하는 optimizer가 있으면 안됨
# 모델이 성능을 잘 내는지 확인은 해야 하므로 정답과 예측값 사이의 간극을 보는 loss_fn은 당연히 있어야 한다.
def evaluate(model, device, x_val_tensor, y_val_tensor, loss_fn):
    with torch.no_grad():
        model.eval()
        x_val_tensor = x_val_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)

        pred = model(x_val_tensor)
        eval_loss = loss_fn(pred, y_val_tensor).item()
        
        return pred, eval_loss
    
# 1 데이터 
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/titanic.3.csv"
data = pd.read_csv(path)
print(data.shape) #(891, 11)
#print(data.head()) 

# 1-1 전처리
target = ["Age","Fare","Sex"]
x = data.drop(labels="Survived",axis=1)
x = data.loc[: , target]
y = data["Survived"]
print(x.shape, y.shape) #(891, 10) (891,)
print(type(x), type(y)) #<class 'pandas.DataFrame'> <class 'pandas.Series'>

# 1-2 가변수화 
x = pd.get_dummies(data=x, columns=["Sex"], drop_first=True)
print(x.head())

# 1-3 데이터셋 분리
x_train, x_val, y_train, y_val = train_test_split(x,y, train_size=0.8, random_state=11, stratify=y)

# 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
print(type(x_train)) # <class 'numpy.ndarray'>

# 0 cpu or gpu ?
if is_available():
    device = "cuda"
else: 
    device = "cpu"
    
# 2 모델구성 
train_DataLoader, x_val_tensor, y_val_tensor = makeDataSet(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, batch_size=32)
model = Sequential(
    Linear(in_features=3, out_features=15),
    ReLU(),
    Linear(15, 1),
    Sigmoid()
).to(device)


# 3 컴파일 및 훈련
loss_fn = BCELoss()
optimizer = Adam(model.parameters(), lr = 0.08)

epochs = 100
train_loss_list, val_loss_list = [], []

for i in range(epochs):
    train_loss_value = train(model=model, device=device, train_DataLoader=train_DataLoader, loss_fn=loss_fn, optimizer=optimizer)    
    _, val_loss_value = evaluate(model=model, device=device, x_val_tensor=x_val_tensor, y_val_tensor=y_val_tensor, loss_fn=loss_fn)

    train_loss_list.append(train_loss_value)
    val_loss_list.append(val_loss_value)
    
    print(f"Epoch{i+1}, train_loss:{np.round(train_loss_value,4)}, val_loss:{np.round(val_loss_value,4)}, device:{device}")


# 4 예측 및 평가 
pred, _= evaluate(model=model, device=device, x_val_tensor=x_val_tensor, y_val_tensor=y_val_tensor, loss_fn=loss_fn)
print(type(pred)) #Pytorch의 Tensor 이다. Pandas의 데이터 타입을 NumPy로 변환할 때는, to_numpy() 혹은 values인데, 
                  #pred는 Pytorch의 Tensor데이터 타입이므로, numpy() 혹은 GPU에 있을 경우 pred.cpu().numpy()를 사용한다.
pred = pred.cpu().numpy()
pred = np.where(pred>=0.5, 1, 0)

print(pred)
