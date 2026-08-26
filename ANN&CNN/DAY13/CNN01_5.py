import tensorflow as tf
from tensorflow.keras import layers
tf.random.set_seed(42)
x = tf.ones((1, 8))
drop = layers.Dropout(0.5)
# ①같은 입력으로 훈련 호출 두 번
train_a = drop(x, training=True)
train_b = drop(x, training=True)
# ② 추론 호출
infer = drop(x, training=False)
print("train A:", train_a.numpy())
print("train B:", train_b.numpy())
print("infer :", infer.numpy())
# ③ 동작 불변조건 확인
assert tf.reduce_any(train_a == 0.)
assert tf.reduce_any(train_b == 0.)
assert not tf.reduce_all(train_a == train_b)
tf.debugging.assert_equal(infer, x)
print("훈련은 무작위, 추론은 원본 유지")
