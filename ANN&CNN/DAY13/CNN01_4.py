import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import  fashion_mnist
from keras.layers import RandomRotation
(x,y),_ = fashion_mnist.load_data() # (x_train, y_train), (x_test, y_test)

# ① 첫 이미지를 0~1, (H,W,C)로 준비
image = x[0].astype("float32") / 255.
image = image[...,None]

# 약한 회전 전용
rotate = RandomRotation(0.08, fill_mode="nearest", seed=42)
aug = rotate(image[None], training=True)[0]
print("label:", int(y[0]))
print("shape:", image.shape, aug.shape)
print("range: ", float(tf.reduce_min(aug)), float(tf.reduce_max(aug)))

# 원본과 증강 결과 비교
for i, (img, title) in enumerate([(image, "원본"), (aug, "회전")]):
    plt.subplot(1, 2, i+1)
    plt.imshow(img[...,0], cmap="gray")
    plt.title(title); plt.axis("off")
plt.show()
assert image.shape == aug.shape
assert 0. <= float(tf.reduce_min(aug)) <= float(tf.reduce_max(aug)) <= 1.0