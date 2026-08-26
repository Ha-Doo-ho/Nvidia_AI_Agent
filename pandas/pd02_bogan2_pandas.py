import numpy as np
import pandas as pd

# 해당 페이지는 매우매우 중요하다. 실무에서 무조건 이것을 시킨다. 10일 한 것 중에서 가장 중요하다. 

data = pd.DataFrame([
    [2, np.nan, 6, 8, 10], 
    [2,4,np.nan,8,np.nan],
    [2,4,6,8,10],
    [np.nan, 4, np.nan, 8, np.nan] 
])
print(data)
data = data.T
data.columns = ['x1','x2', 'x3', 'x4']
print(data) #(5,4)

# pandas는 Series, DataFrame 단 두가지이다. 


#0. 결측치 확인
#print(data.isnull()) #판다스에 있는 데이터는 numpy이다. 그래서 numpy를 이용이 가능하다. 
#print(data.isnull().sum())

#열의 결측치 개수를 알 수 있다. 
# x1    1
# x2    2
# x3    0
# x4    3

# 이상치도 결측치 처리 방식과 똑같이 처리할 수 있다. 데이터 조작의 위험을 감수하고 해야 한다. 
#print(data.info()) # null 찾는 다른 방법이다. 


# 1. 결측치 삭제 
print(data.dropna()) #Nan이 하나라도 있는 "행"은 전부 지운다. 실무에서는 데이터가 매우 많아서 모델을 빨리 만드려면 이러한 경우가 있다. 
print(data.dropna(axis=0)) # 디폴트이다. 행별로 Nan값이 있는 것은 전부 지운다. 
print(data.dropna(axis=1))

#2-1. 특정값 - 평균
means = data.mean() # pandas에서는 알아서 자동으로 컬럼별로 평균을 내줌 
print(means) 
data2 = data.fillna(means)
print(data2) #컬럼별 평균치로 채운다. 

# 2-2. 특정값 - 중위값.
med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

# 2-3. 특정 값 - 0
data4 = data.fillna(0)
print(data4)

# 2-4. 특정값 - 777
data5 = data.fillna(777)
print(data5)

# 2-5 특정값 - ffill  #시계열 데이터에서는 성능이 정말 좋다. 특히 분단위로 특정한 데이터에서 ffill, bfill은 정말 성능이 좋다. 
data6 = data.ffill()
print(data6)

# 2-6 특정값 - bfill
data7 = data.bfill() #시계열 데이터에서는 성능이 정말 좋다. 특히 분단위로 특정한 데이터에서 ffill, bfill은 정말 성능이 좋다. 
print(data7)

# ffill을 사용할 때 만약 가장 "앞"의 데이터가 Nan이면 그대로 Nan이다. 
# bfill을 사용할 때 만약 가장 "뒤"의 데이터가 Nan이면 그대로 Nan이다. 
# 그래서 ffill, bfill을 사용할 때는 가장 앞, 가장 뒤의 데이터가 Nan인지를 확인해야 한다. 

# 뭐가 좋다 알 수 없다. 짬밥으로 어떤 것이 더 좋은지 경험으로 알아야 한다. 
############################################################

"""
지금까지 한 것은 컬럼 전체에 적용되었다. 어떤 컬럼은 평균값, 어떤 컬럼은 중앙값, 어떤 컬럼은 특정값, 어떤 컬럼은 ffill, bfill이 좋을 수 있다. 
그래서 컬럼별로 적용해야 한다. 그것을 시도할 것이다 .
"""

# means = data['x1'].mean()
# print(means)

# med = data['x4'].median()
# print(med) 

# 실습
"""
x1 : median
x2 : ffill
x4 : mean
    
"""

median = data['x1'].median() # 저 x1은 컬럼이다. 판다스는 동작 방식이 매우 직관적이다. 
mean = data['x4'].mean()

data['x1'] = data['x1'].fillna(median)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(mean)
print(data) 

"""
Pandas의 기본 인덱싱 구조는 열(Column) 중심으로 설계되어 있습니다. 
대괄호 []를 사용해 데이터를 선택할 때 가장 먼저 반응하는 것이 열 이름이기 때문입니다.

열 선택: df['열이름'] (열 이름을 바로 적음)

행 선택: 판다스에서는 행선택을 df.loc[행인덱스] 또는 df.iloc[행번호] (별도의 라벨/위치 지정 메서드를 사용해야 함
"""