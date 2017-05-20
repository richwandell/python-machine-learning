try:
    from scipy import optimize
except:
    pass
import numpy as np
import random
import datetime


class NeuralNet:
    def __init__(self, in_size, hidden_size, out_size, learning_rate=1e-5, verbose=False, tol=0.001,
                 max_epoch=False, epoch_report=10, adaptive_learning=False, max_cost=False):
        self.input_layer_size = in_size + 1
        self.hidden_layer_size = hidden_size
        self.output_layer_size = out_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.tol = tol
        self.max_epoch = max_epoch
        self.epoch_report = epoch_report
        self.adaptive_learning = adaptive_learning
        self.max_cost = max_cost

        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        self.backwards = False

    def get_params(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def get_plain_params(self):
        return self.W1, self.W2

    def set_plain_params(self, W1, W2):
        self.W1 = W1
        self.W2 = W2

    def set_params(self, params):
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(params[W1_start:W1_end], (self.input_layer_size, self.hidden_layer_size))

        W2_end = W1_end + self.hidden_layer_size * self.output_layer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        oneMinus = (1 - sig)
        mul = sig * oneMinus
        return mul

    def tanh(self, z):
        return (2 / (1 + np.exp(-2 * z))) - 1

    def relu(self, z):
        return z * (z > 0)

    def relu_prime(self, z):
        rp = np.where(z < 0.0, np.zeros(z.shape), np.ones(z.shape))
        return rp

    def forward(self, X):
        self.Z2 = np.dot(X, self.W1)
        self.A2 = self.sigmoid(self.Z2 + 1)
        self.Z3 = np.dot(self.A2, self.W2)
        last = self.sigmoid(self.Z3 + 1)
        return last

    def find_nearest_vector(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.append(X, [1])
        else:
            x1 = np.ones(X.shape[0])
            X = np.column_stack((X, x1))

        out = self.forward(X)
        for i, ind in enumerate(out):
            out[i] = self.find_nearest_vector(self.possible_y, ind)

        return out

    def cost_function(self, X, y):
        self.y_hat = self.forward(X)
        J = sum((y - self.y_hat) ** 2) / X.shape[1]
        return J

    def cost_function_prime(self, X, y):
        self.y_hat = self.forward(X)
        ymyh = y - self.y_hat
        rsub = (-1) * ymyh

        delta3 = self.sigmoid_prime(self.Z3)

        delta3 = np.multiply(rsub, delta3)
        djdw2 = np.dot(self.A2.T, delta3)
        d3mw2 = np.dot(delta3, self.W2.T)

        sig = self.sigmoid_prime(self.Z2)
        delta2 = d3mw2 * sig
        xtrans = X.T
        djdw1 = np.dot(xtrans, delta2)

        return djdw1, djdw2

    def compute_gradients(self, X, y):
        djdw1, djdw2 = self.cost_function_prime(X, y)
        return np.concatenate((djdw1.ravel(), djdw2.ravel()))

    def __cost_function_wrapper(self, params, X, y):
        self.set_params(params)
        cost = self.cost_function(X, y)
        grad = self.compute_gradients(X, y)
        return cost, grad

    def __bfgs_callback(self, params):
        self.set_params(params)

    def train_optimize_bfgs(self, X, y):
        self.possible_y = np.unique(y)
        x1 = np.ones(X.shape[0])
        X = np.column_stack((X, x1))
        params0 = self.get_params()

        options = {'maxiter': 3000, 'disp': True}
        optimize.minimize(self.__cost_function_wrapper, params0, jac=True, method='L-BFGS-B', args=(X, y),
                          options=options, callback=self.__bfgs_callback)

    def __get_random_batch(self, x_size, batch_size):
        self.backwards = not self.backwards
        start = random.randint(0, x_size)
        end = start + batch_size
        if end > x_size:
            end = x_size
        if x_size < self.batch_size:
            start = 0
            end = x_size
        if self.backwards:
            return end, start

        return start, end

    def train_sgd(self, X, y):
        self.possible_y = np.unique(y)
        x1 = np.ones(X.shape[0])
        X = np.column_stack((X, x1))
        alpha = self.learning_rate
        epoch = 0
        last_cost = []

        while not self.max_epoch or epoch < self.max_epoch:
            start = datetime.datetime.now()
            s = np.arange(X.shape[0])
            np.random.shuffle(s)

            for i in np.nditer(s, op_flags=['readonly']):
                x = X[i].reshape((1, X.shape[1]))
                djdw1, djdw2 = self.cost_function_prime(x, y[i])
                W1, W2 = self.get_plain_params()
                d1 = (djdw1 * alpha)
                W1 = W1 - d1
                d2 = (djdw2 * alpha)
                W2 = W2 - d2
                self.set_plain_params(W1, W2)

            epoch += 1
            cost = self.cost_function(X, y)
            cost = sum(cost)

            if self.verbose and epoch % self.epoch_report == 0:
                print("Epoch: " + str(epoch) + " Cost: " + str(cost))
                end = datetime.datetime.now()
                seconds = (end - start).total_seconds()
                print("Epoch Time: milliseconds: " + str(seconds * 1000) + " seconds: " + str(seconds))

            if len(last_cost) == 4:
                last_cost.pop(0)
            last_cost.append(cost)
            diff = sum([abs(cost - x) for x in last_cost])

            if len(last_cost) == 4 and diff < self.tol and self.max_cost is False:
                if self.adaptive_learning:
                    alpha = 1.05 * alpha
                    last_cost = []
                    print("Increasing Learning Rate")
                else:
                    print("Precision level: " + str(self.tol) + " reached")
                    break

            if self.max_cost and cost < self.max_cost:
                print("Cost level: " + str(self.max_cost) + " reached")
                break
