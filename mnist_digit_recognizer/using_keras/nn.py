import keras
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn import metrics


df = pd.read_csv("../data/train.csv.gz")
y = keras.utils.to_categorical(df[df.columns[0]], 10)
X = np.array(df[df.columns[1:]], dtype=float).reshape(df.shape[0], 784) / 255

train_size = int(X.shape[0] * .75)

pca = PCA(n_components=200)
training_data = pca.fit_transform(X[:train_size], y[:train_size])

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(200,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
loss = model.fit(training_data, y[:train_size], batch_size=1500, epochs=20, verbose=1)

# get predictions for the data not used in the classifier
predicted = np.argmax(model.predict(pca.transform(X[train_size:])), axis=1)
actual = np.argmax(y[train_size:], axis=1)
print(metrics.classification_report(actual, predicted))
print(metrics.confusion_matrix(actual, predicted))

joblib.dump(pca, '../trained/keras_pca.pkl')
model.save('../trained/keras_neural_network.h5')




