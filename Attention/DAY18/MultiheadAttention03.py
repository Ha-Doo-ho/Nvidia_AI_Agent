import numpy as np

padding_mask = np.array([
    [False, False, True, True, True]
])

print(padding_mask.shape) # (1, 5)
print(padding_mask)       # [[False False  True  True  True]]

query_mask = np.expand_dims(padding_mask, axis=2)
print(query_mask.shape) # (1, 5, 1)
print(query_mask) 
"""
[[[False]
  [False]
  [ True]
  [ True]
  [ True]]]
"""
mask = np.array([False, False, True, True, True])
expand_dims_mask = np.expand_dims(mask, axis=1)  
print(expand_dims_mask.shape) #(5, 1)
print(expand_dims_mask) 
# [[False]
#  [False]
#  [ True]
#  [ True]
#  [ True]]

# 결론: 차원 상관 없이 가로로 펼쳐져 있어도 expand_dims를 하면 세로로 펴진다. 