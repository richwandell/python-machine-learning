import numpy as np
from numpy_mlp.NeuralNet import NeuralNet

X = np.array(([3, 5], [5, 1], [10, 2], [1, 24]), dtype=float)
y = np.array(([75], [82], [93], [10]), dtype=float)

# Normalize
m = np.amax(X, axis=0)
X = X / m
y = y / 100  # Max test score is 100

n = NeuralNet(2, 3, 1, learning_rate=0.3, verbose=True, epoch_report=1000, tol=1e-8, max_cost=0.0001)
n.train_sgd(X, y)

y_hat = n.predict(X[0])
print(y_hat)

y_hat = n.predict(X[1])
print(y_hat)

y_hat = n.predict(X[2])
print(y_hat)

y_hat = n.predict(X[3])
print(y_hat)
