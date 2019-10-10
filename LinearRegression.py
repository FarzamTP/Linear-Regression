from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, file_path, theta, num_iter, learning_rate, verbose):
        self.theta = theta
        self.file_path = file_path
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.X, self.y = self.read_data()
        self.verbose = verbose
        self.cost_history = []

    def plot_data(self):
        plt.plot(self.X[:, 0], self.X[:, 1], 'o')
        plt.xlabel('Data')
        plt.ylabel('Label')
        plt.show()

    def plot_line(self):
        x = np.linspace(5, 25)
        y = self.theta[0] + self.theta[1] * x
        plt.plot(self.X[:, 0], self.X[:, 1], 'o')
        plt.plot(x, y, '-r')
        plt.xlabel('Data')
        plt.ylabel('Label')
        plt.show()

    def read_data(self):
        X = []
        y = []
        with open('data.txt') as f:
            for line in f.readlines():
                X_1 = float(line.replace('\n', '').split(',')[0])
                X_2 = float(line.replace('\n', '').split(',')[1])
                y.append(X_2)
                x = (X_1, X_2)
                X.append(x)
        return np.asarray(X), np.asarray(y)

    def calculate_cost(self):
        m = self.y.shape[0]
        h = np.matmul(self.X, np.transpose(self.theta))
        J = (1 / 2 * m) * sum((h - self.y) ** 2)
        return J

    def train(self):
        for i in range(self.num_iter):
            m = self.y.shape[0]
            h = np.matmul(self.X, self.theta)
            self.theta[0] -= self.learning_rate * (1 / m) * sum(h - self.y)
            self.theta[1] -= self.learning_rate * (1 / m) * sum((np.dot(h - self.y, self.X)))
            self.cost_history.append(self.calculate_cost())
            if self.verbose:
                print("Batch: ", i + 1)
                print("Theta: ", self.theta)
                print("Cost: ", self.calculate_cost())

    def plot_cost(self):
        plt.plot(self.cost_history[10:], 'o')
        plt.xlabel('#Iteration')
        plt.ylabel('Cost')
        plt.show()

    def compare_to_Sklearn(self):
        model = LR()
        model.fit(self.X, self.y)
        print("My LR Theta:", self.theta)
        print("Sklearn LR Theta:", model.coef_)