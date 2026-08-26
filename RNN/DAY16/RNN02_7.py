import tensorflow as tf
from keras.layers import GRU 
x = tf.zeros((2, 4, 3))
# GRU는 별도의 cell state C를 반환하지 않습니다.
output, h = GRU(5, return_state=True)(x)
print("output:", output.shape)
print("h :", h.shape)
print("output==h:", tf.reduce_all(output == h).numpy())