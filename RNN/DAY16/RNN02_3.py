from keras.layers import TextVectorization
texts = ["good movie", "bad film", "not good"]
vec = TextVectorization(
max_tokens=8, output_sequence_length=4)
vec.adapt(texts)
vocab = vec.get_vocabulary()
ids = vec(["good story"])
print("vocab:", vocab)
print("ids :", ids.numpy())
assert vocab[:2] == ["", "[UNK]"]
assert ids.shape == (1, 4)
assert ids.numpy()[0, -1] == 0
