import numpy as np
values = np.array([
    [10, 20, 15, 14],
    [15, 12, 10, 11]
], dtype=np.float32)

X = values[...,np.newaxis]
expected = (2, 4, 1)

print("values:", values.shape)
print("RNN X :", X.shape)
assert X.shape == expected
print("shape check: PASS")