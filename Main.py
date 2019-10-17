from LinearRegression import LinearRegression
import numpy as np


# reads data from file and splits it into X, y.
def read_data(file_path):
    X = []
    y = []
    with open(file_path) as f:
        for line in f.readlines():
            X_1 = float(line.replace('\n', '').split(',')[0])
            X_2 = float(line.replace('\n', '').split(',')[1])
            y.append(X_2)
            x = (X_1, X_2)
            X.append(x)
    return np.asarray(X), np.asarray(y)


# calling read_data function and returns X, y
X, y = read_data('data.txt')

# declares model as follows.
model = LinearRegression(X, y, theta=[0, 0], num_iter=1000, learning_rate=0.001, verbose=True)

# plotting data
model.plot_data()

# starting to train data
model.train(X, y)

# plots cost history.
model.plot_cost()

# plots decision boundary.
model.plot_line()

# compares model to sklearn.linear model.
print()
model.compare_to_Sklearn()
