import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn import metrics
from scipy.interpolate import spline
import matplotlib.pyplot as plt

df = pd.read_csv('../data/training.csv.gz')
df = df.dropna()
df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
train_size = df['Image'].count() #int(df['Image'].count() * .75)

X = np.vstack(df['Image'].values)
y = df[df.columns[:-1]].values.astype(np.float)
X = X / 255
y = (y - 48) / 48

if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, 96, 96)
    input_shape = (1, 96, 96)
else:
    X = X.reshape(X.shape[0], 96, 96, 1)
    input_shape = (96, 96, 1)

# model = Sequential()
#
# model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(30))

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(30))

model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
loss = model.fit(X[:train_size], y[:train_size], epochs=40, verbose=1, validation_split=0.2)

# test_rows = X[train_size:train_size + 16]
# y_pred = model.predict(test_rows)


# fig = plt.figure()
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(len(test_rows)):
#     sub = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
#     img = np.reshape(test_rows[i], (96, 96))
#     sub.imshow(img, cmap='gray')
#     sub.scatter(y_pred[i][0::2] * 48 + 48, y_pred[i][1::2] * 48 + 48, marker='x', s=10)
#
# plt.show()

model.save('../trained/keras_conv_neural_network.h5', include_optimizer=False)


# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=X.shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(500))
# model.add(Activation('relu'))
# model.add(Dense(500))
# model.add(Activation('relu'))
# model.add(Dense(30))
