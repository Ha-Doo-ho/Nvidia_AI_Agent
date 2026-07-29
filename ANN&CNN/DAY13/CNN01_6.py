import numpy as np
import tensorflow as tf 
from keras.layers import BatchNormalization

x = tf.constant([[10.,12.],
                 [20.,24.],
                 [30.,36.], 
                 [40.,48.]])

# ① 학습 모드: 현재 배치 통계 사용
bn = BatchNormalization(center=False, scale=False)
y = bn(x, training=True)
mean = tf.reduce_min(y, axis=0)
std = tf.math.reduce_std(y, axis=0)
print("input shape:", x.shape)
print("output shape: ", y.shape)
print("mean:", mean.numpy())
print("std: ", std.numpy())
assert x.shape == y.shape
np.testing.assert_allclose(mean, [0,0], atol=.01)
np.testing.assert_allclose(std, [1., 1.], atol=.01)

