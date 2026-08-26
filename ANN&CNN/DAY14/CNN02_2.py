import numpy as np
feature_maps = np.array([[
    [[0.,1.,0.], [0.,2.,0.]],
    [[0.,3.,1.], [0.,4.,1.]]
]])

print(feature_maps.shape)
strength = np.mean(np.abs(feature_maps), axis=(0, 1, 2))

print("strength", strength)

top = np.argsort(strength)[::-1] #가장 작은 것부터 오름차순으로 정렬하는데, args로 보인다. 그런데, [::-1]을 걸었으니 가장 큰 것부터 오게 된다. 
print("shape:", feature_maps.shape)
print("strength: ", strength)
print("top: ", top) 