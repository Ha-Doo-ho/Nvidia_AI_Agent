from tensorflow import constant
from keras.layers import TextVectorization
import numpy as np

texts = constant(np.array([
    "good movie",
    "this movie is very good today"
]))

vec = TextVectorization(output_mode="int", output_sequence_length=4) #output_sequence_length이게 토큰 개수이다. 
vec.adapt(texts)
print(vec(texts).numpy())