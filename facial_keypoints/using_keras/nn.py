import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

df = pd.read_csv('../data/training.csv.gz')
df = df.dropna()
df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
train_size = int(df['Image'].count() * .75)

X = np.vstack(df['Image'].values)
y = df[df.columns[:-1]].values.astype(np.float)
X = X / 255
y = (y - 48) / 48

model = Sequential()
model.add(Dense(100, input_shape=X[0].shape, activation='relu', use_bias=True))
model.add(Dropout(0.1))
model.add(Dense(30))

model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
loss = model.fit(X[:train_size], y[:train_size], epochs=40, verbose=1, validation_split=0.2)

test_rows = X[train_size:train_size + 16]
y_pred = model.predict(test_rows)


fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(len(test_rows)):
    sub = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    img = np.reshape(test_rows[i], (96, 96))
    sub.imshow(img, cmap='gray')
    sub.scatter(y_pred[i][0::2] * 48 + 48, y_pred[i][1::2] * 48 + 48, marker='x', s=10)

plt.show()




