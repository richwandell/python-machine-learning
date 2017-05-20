import csv, io, gzip
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.externals import joblib
from numpy_mlp.NeuralNet import NeuralNet

# Setup X and y
X, y = [], []
with io.TextIOWrapper(gzip.open("../mnist_digit_recognizer/data/train.csv.gz", "r")) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(csvreader):
        if i == 0: continue
        vals = row[1:]
        X.append(vals)
        y.append([row[0]])

# Normalize
X = np.array(X, dtype=float)
X = X / 255
y = np.array(y, dtype=float)
y = y / 10


# select training data size using 75% of the source
train_size = int(X.shape[0] * .75)
# perform principle component analysis and use 50 feature
pca = PCA(n_components=50)
training_data = pca.fit_transform(X[:train_size], y[:train_size])


n = NeuralNet(50, 30, 1, learning_rate=0.1, verbose=True, epoch_report=5, tol=0.09,
              max_epoch=200, max_cost=0.1, adaptive_learning=True)
n.train_sgd(training_data, y[:train_size])

# get predictions for the data not used in the classifier
predicted = np.array((n.predict(pca.transform(X[train_size:])) * 10), dtype=int)
actual = np.array(y[train_size:] * 10, dtype=int)
print(metrics.classification_report(actual, predicted))
print(metrics.confusion_matrix(actual, predicted))

joblib.dump(n, 'trained/my_neural_network.pkl')
#0
print(y[1])
y_hat = n.predict(pca.transform(np.atleast_2d(X[1])))
print(y_hat)

# 1
print(y[0])
y_hat = n.predict(pca.transform(np.atleast_2d(X[0])))
print(y_hat)

# 2
print(y[16])
y_hat = n.predict(pca.transform(np.atleast_2d(X[16])))
print(y_hat)

# 3
print(y[9])
y_hat = n.predict(pca.transform(np.atleast_2d(X[9])))
print(y_hat)

