import tensorflow as tf
from tensorflow.keras import layers
x = tf.constant([[[1.], [2.], [0.], [0.]],
[[1.], [2.], [3.], [4.]]])
masking = layers.Masking(mask_value=0.0)
masked_x = masking(x)
mask = masking.compute_mask(x)
rnn = layers.SimpleRNN(3)
masked_out = rnn(masked_x)
short_out = rnn(x[:1, :2, :])
same = tf.reduce_all(tf.abs(masked_out[:1] - short_out) < 1e-6)
print("mask:", mask.numpy())
print("output:", masked_out.shape, bool(same.numpy()))
