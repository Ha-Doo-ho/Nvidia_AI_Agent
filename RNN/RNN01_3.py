import numpy as np
train = np.array([10, 12, 14], dtype=np.float32)
val = np.array([16, 18], dtype=np.float32)
mu, sigma = train.mean(), train.std()
train_z = (train - mu) / sigma #[-1.225, 0, 1.225]
val_z = (val - mu) / sigma #[2.449, 3.674]
print("mu, sigma:", round(float(mu), 3),
round(float(sigma), 3))
print("train mean:", round(float(train_z.mean()), 6))
print("val mean :", round(float(val_z.mean()), 3))
assert np.isclose(train_z.mean(), 0.0)
assert not np.isclose(val_z.mean(), 0.0)