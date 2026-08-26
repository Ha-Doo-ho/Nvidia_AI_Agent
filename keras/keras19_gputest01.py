import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU") #GPU
print(gpus)

if(gpus):
    print("GPU 돈다~")
else:
    print("GPU 없다~")
    
    #[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    # 이렇게 뜬다. GPU를 사용할 수 있다는 뜻이다. GPU: 0이란 뜻은 0번 그래픽 카드가 활성화된다는 뜻이다. 
    
    # 넘파이도 1.x 버전으로바꿔야 함.
    # 모든 프로그램이 넘파이를 사용하는데, 그 프로그램들이 요구하는 넘파이 버전이 천차만별이기 때문에
    # 넘파이에서는 버전 관련된 에러가 매우매우 많이 발생하게 된다. 방법은 롤벡(이전 버전으로 돌아가는 것) 혹은 더 상위 버전 
    