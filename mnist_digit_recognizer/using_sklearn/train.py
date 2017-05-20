import gzip, io, csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import metrics


X, y = [], []
with io.TextIOWrapper(gzip.open("../data/train.csv.gz", "r")) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(csvreader):
        if i == 0: continue
        vals = row[1:]
        X.append(vals)
        y.append([row[0]])


X = np.array(X, dtype=float)
X = X / 255
y = np.array(y, dtype=str)

# plot the first 10 images


def gimg(i):
    # images must be 28x28x3
    return np.reshape(
        # greyscale images using the same value for R G B
        np.column_stack(
            (X[i], X[i], X[i])
        ),
        (28, 28, 3)
    )


img = gimg(0)
for i in range(1, 5):
    img = np.column_stack((img, gimg(i)))

img1 = gimg(6)
for i in range(7, 11):
    img1 = np.column_stack((img1, gimg(i)))

img = 1 - np.row_stack((img, img1))

plt.imshow(img)
plt.show()

# select training data size using 75% of the source
train_size = int(X.shape[0] * .75)
# perform principle component analysis and use 50 feature
pca = PCA(n_components=50)
training_data = pca.fit_transform(X[:train_size], y[:train_size])
# create simple neural network and train
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu', max_iter=3000,
                    hidden_layer_sizes=(30,), random_state=1)
clf.fit(training_data, y[:train_size].ravel())

# get predictions for the data not used in the classifier
predicted = clf.predict(pca.transform(X[train_size:]))
actual = y[train_size:]
print(metrics.classification_report(actual, predicted))
print(metrics.confusion_matrix(actual, predicted))

joblib.dump(pca, '../trained/sklearn_pca.pkl')
joblib.dump(clf, '../trained/sklearn_neural_network.pkl')


