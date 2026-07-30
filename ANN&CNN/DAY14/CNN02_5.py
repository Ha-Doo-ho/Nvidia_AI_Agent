import tensorflow as tf
maps = tf.constant([[
[[1., 0.], [0., 1.]],
[[2., 1.], [1., 0.]],
]])
print(maps.shape) #(1, 2, 2, 2)
weights = tf.constant([.8, -.2])
cam = tf.reduce_sum(
maps[0] * weights, axis=-1
)
heatmap = tf.maximum(cam, 0)
heatmap /= tf.reduce_max(heatmap) + 1e-8
print("cam:\n", cam.numpy())
print("heatmap:\n", heatmap.numpy())
