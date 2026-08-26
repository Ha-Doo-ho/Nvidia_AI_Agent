import numpy as np
import tensorflow as tf 
from keras.callbacks import EarlyStopping

val_loss = np.array([.62, .51, .45, .46, .47, .48])

stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

best_epoch = int(val_loss.argmin() + 1)
waited = len(val_loss) - best_epoch
print("monitor:", stop.monitor)
print("patience:", stop.patience)
print("restore:", stop.restore_best_weights)
print("best epoch:", best_epoch, "/ waited:", waited)
assert stop.monitor == "val_loss"
assert stop.patience == 3
assert stop.restore_best_weights is True
assert best_epoch == 3 and waited == 3

