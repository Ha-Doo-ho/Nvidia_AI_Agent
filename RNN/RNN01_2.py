import numpy as np
series = np.array([10, 12, 15, 14, 18])
W = 2
samples = []
for i in range(len(series) - W):
    X = series[i:i + W]
    y = series[i + W]
    samples.append((X, y))
    print(f"X={X} -> y={y}")

assert len(samples) == len(series) - W
print("sample count:", len(samples))
