import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # weights + bias

    def activation(self, x):
        return 1 if x >= 0 else -1  # 1 = Popular, -1 = Not Popular

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def fit(self, X, y):
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
