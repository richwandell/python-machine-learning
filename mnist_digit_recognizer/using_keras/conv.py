import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn import metrics
from scipy.interpolate import spline
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.bloss, self.bacc = [], []
        self.batch = 0
        self.batch_interval = 10

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            self.batch += 1
            if self.batch % self.batch_interval == 0:
                self.bloss.append(logs.get('loss'))
                self.bacc.append(logs.get('acc'))


df = pd.read_csv("../data/train.csv.gz")
y = keras.utils.to_categorical(df[df.columns[0]], 10)
X = np.array(df[df.columns[1:]], dtype=float) / 255

train_size = int(X.shape[0] * 1)

if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X = X.reshape(X.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# 384s - loss: 0.2169 - acc: 0.9357
model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', input_shape=input_shape))
model.add(Dropout(0.25))
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = LossHistory()
model.fit(X[:train_size], y[:train_size], batch_size=200, epochs=20, verbose=1, callbacks=[history])

plot_blx = np.array(range(0, len(history.bloss))) * history.batch_interval
plt.figure(1)
plt.subplot(221)
plt.xlabel('Batch Num')
plt.ylabel('Loss')
plt.plot(plot_blx, history.bloss)

plt.subplot(222)
plt.xlabel('Batch Num')
plt.ylabel('Accuracy')
plt.plot(plot_blx, history.bacc)

plt.show()

# get predictions for the data not used in the classifier
# predicted = np.argmax(model.predict(X[train_size:]), axis=1)
# actual = np.argmax(y[train_size:], axis=1)
# print(metrics.classification_report(actual, predicted))
# print(metrics.confusion_matrix(actual, predicted))

model.save('../trained/keras_conv_neural_network.h5', include_optimizer=False)





