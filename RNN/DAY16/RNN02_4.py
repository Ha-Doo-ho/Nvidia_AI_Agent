import tensorflow as tf
from keras.layers import Embedding
ids = tf.constant([
[2, 3, 0],
[4, 2, 1]
])
embedding = Embedding(
input_dim=10, output_dim=4, mask_zero=True)
vectors = embedding(ids)
print(ids.shape, "->", vectors.shape)