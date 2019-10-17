from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, X, y, theta, num_iter, learning_rate, verbose):
        self.X = X
        self.y = y
        self.theta = theta
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.cost_history = []

    # plots data points.
    def plot_data(self):
        plt.plot(self.X[:, 0], self.X[:, 1], 'o')
        plt.xlabel('Data')
        plt.ylabel('Label')
        plt.show()

    # plots decision boundary over data.
    def plot_line(self):
        x = np.linspace(5, 25)
        y = self.theta[0] + self.theta[1] * x
        plt.plot(self.X[:, 0], self.X[:, 1], 'o')
        plt.plot(x, y, '-r')
        plt.xlabel('Data')
        plt.ylabel('Label')
        plt.show()

    # calculates cost function
    def calculate_cost(self):
        m = self.y.shape[0]
        h = np.matmul(self.X, np.transpose(self.theta))
        J = (1 / 2 * m) * sum((h - self.y) ** 2)
        return J

    # training model.
    def train(self, X, y):
        m = self.y.shape[0]
        for i in range(self.num_iter):
            h = np.matmul(X, self.theta)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient
            self.cost_history.append(self.calculate_cost())
            if self.verbose:
                print("Batch: ", i + 1)
                print("Theta: ", self.theta)
                print("Cost: ", self.calculate_cost())

    # plots cost history.
    def plot_cost(self):
        plt.plot(self.cost_history, 'o')
        plt.xlabel('#Iteration')
        plt.ylabel('Cost')
        plt.show()

    # makes a sklearn model and prints its Theta.
    def compare_to_Sklearn(self):
        model = LR()
        model.fit(self.X, self.y)
        print("My LR Theta:", self.theta)
        print("Sklearn LR Theta:", model.coef_)