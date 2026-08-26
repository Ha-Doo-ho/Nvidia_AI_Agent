import tensorflow as tf
heatmap = tf.constant([
[0.0, 0.8],
[0.2, 1.0],
])
big = tf.image.resize(
heatmap[None, ..., None],
(6, 6),
method="bilinear",
)[0, ..., 0]
print(heatmap.shape, "→", big.shape)
print(float(tf.reduce_min(big)),
float(tf.reduce_max(big)))