from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense

#모델 구성
cnn_model = Sequential(name="cnn_baseline")
cnn_model.add(Input(shape=(28,28,1), name="Input_layer"))
cnn_model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation='relu', name="conv_1"))
cnn_model.add(MaxPool2D(pool_size=2)) # strides를 적용하지 않는 이유는 어차피 pool_size에 맞게 돌아야 하는게 맞기 때문이다. 
                                        # 만약 pool_size = 2x2 크기라면 strides는 2칸이 되어야 모든 칸을 전부 확인할 것이다. 
                                        # 그것이 배운대로 가는 것이니 맞음. 
                                        # 패딩도 있다. 0을 채우는 것인데, 굳이 플링에서 0을 채운다.? 이것도 배운 것에 없음. 전에 배웠을 때도 풀링에서는 굳이 패딩을 하지 않았음.   
                                        # 즉 패딩을 하지 않는 "valid"가 들어가는 것이 이전, 지금 배운대로 하면 맞음. 여기서 Padding을 할 이유는 없음
cnn_model.add(Conv2D(filters=64, kernel_size=3, padding="same", name="conv_2"))  # 커널 즉, 돋보기의 크기는 보통 홀수이다. 왜? 홀수 x 홀수 는 정중앙이 있어서 공간적인 대칭성이 유지된다. 왜곡현상을 막을 수 있다. 
cnn_model.add(MaxPool2D(pool_size=2, name="pool_2"))
cnn_model.add(Flatten(name="flatten"))
cnn_model.add(Dense(64, activation='relu', name="hidden")) # 요즘은 Dense가 마지막 출력에만있다. 그래도 하나 정도는 있다. 
cnn_model.add(Dense(10, activation="softmax",)) # 마지막 출력은 클래스의 개수


#컴파일 및 훈련
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

