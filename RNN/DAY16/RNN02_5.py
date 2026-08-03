import tensorflow as tf
from keras.layers import Embedding
ids = tf.constant([
[2, 3, 0, 0],
[4, 2, 1, 0]
])
embedding = Embedding(
input_dim=10, output_dim=4, mask_zero=True)
mask = embedding.compute_mask(ids) #마스크가 제대로 만들어졌는지 "확인"하는 용도로 사용한다. 그 용도가 아니면 호출하는 경우는 없음. 
print(mask.numpy())