import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input((8, 8, 1))
f = layers.Conv2D(4, 3, activation="relu")(inputs)
scores = layers.Dense(3)(
layers.GlobalAveragePooling2D()(f)
)
model = keras.Model(inputs, scores)
grad_model = keras.Model(model.inputs, [f, scores])
img = tf.ones((1, 8, 8, 1))
with tf.GradientTape() as tape:
    maps, scores = grad_model(img, training=False)
    target_score = scores[:, 0]
grads = tape.gradient(target_score, maps)
print(maps.shape, grads.shape)
