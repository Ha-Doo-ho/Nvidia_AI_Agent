import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Embedding, Add, MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D  
from keras_hub.layers import PositionEmbedding, TokenAndPositionEmbedding
from keras import ops
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import time 
import matplotlib.pyplot as plt 

# 1 데이터
VOCAB = 10000
MAXLEN = 200
EMBEDDING_DIM = 128
PATH = "./_save/keras_Attention_imdb3.keras"

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)

# 1-2 데이터 전처리
x_test = pad_sequences(sequences=x_test, maxlen=MAXLEN, padding="pre", truncating="pre")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=11, shuffle=True, stratify=y_train)
x_train = pad_sequences(sequences=x_train, maxlen=MAXLEN, padding="pre", truncating="pre")
x_val = pad_sequences(sequences=x_val, maxlen=MAXLEN, padding="pre", truncating="pre",)


# 2 모델 구성
inputs = Input(shape=(200, ))

TP_Embedding= TokenAndPositionEmbedding(vocabulary_size=VOCAB, sequence_length=MAXLEN, embedding_dim=EMBEDDING_DIM, mask_zero=True)(inputs)

attention_output = MultiHeadAttention(num_heads=3, key_dim=64)(query=TP_Embedding, key=TP_Embedding, value=TP_Embedding,)

x = GlobalAveragePooling1D()(attention_output,)

x = Dense(64, activation="relu")(x)

x = Dropout(0.3)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)

# 3 컴파일 및 훈련
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy", AUC()])

es = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
mcp = ModelCheckpoint(filepath=PATH, monitor="val_loss", save_best_only=True, mode="min",)

start_time = time.time()
model.fit(x_train, y_train, batch_size=16, epochs=10, callbacks=[es, mcp])
end_time = time.time()

real_time = np.round(end_time-start_time, 4)

# 4 평가 및 예측
loss, accuracy, auc = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict(x_test).flatten()
y_predict_class = (y_predict >= 0.5).astype("int32")
print(f"정확도: {round(accuracy, 4)}, 소요시간: {real_time}")

cm = confusion_matrix(y_test, y_predict_class)
labels = ["Negative", "Positive"]

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cmd.plot(cmap="Oranges_r", xticks_rotation=45)
plt.show()


