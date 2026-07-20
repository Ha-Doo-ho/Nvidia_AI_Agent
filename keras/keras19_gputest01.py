import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

if(gpus):
    print("GPU 돈다~")
else:
    print("GPU 없다~")
    #넘파이도 1.x 버전으로바꿔야 함.