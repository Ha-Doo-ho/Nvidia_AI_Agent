import tensorflow as tf
from keras.layers import SimpleRNN, LSTM, GRU
x = tf.zeros((2, 12, 4))
models = [SimpleRNN(8), LSTM(8), GRU(8)]
params = []
for layer in models:
    y = layer(x)
    params.append(layer.count_params())
    print(layer.__class__.__name__, y.shape,
        layer.count_params())
assert params == [104, 416, 336]
    