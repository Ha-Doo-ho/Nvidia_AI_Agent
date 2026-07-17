import numpy as np 

# 여기서의 핵심은 .shape로 확인하는 습관을 무조건 길러라. 실무 필수입니다. 
# 왜? 데이터 열을 확인해야 맞추어야 input_dim 혹은 input_shape 에 맞추어야 하기 때문이다.
x1 = np.array([1,2,3])
print("x1 =", x1.shape) # x1 = (3,)  벡터

x2 = np.array([[1,2,3]])
print("x2 =", x2.shape) # x1 = (1, 3)   행렬

x3 = np.array([[1,2,3], [4,5,6]])
print("x3 =", x3.shape) # x1 = (2, 3)

x5 =np.array([[[[1]]], [[[2]]]])
print('x5 =', x5. shape) # x1 = (2,1,1,1)






