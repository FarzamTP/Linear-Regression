from LinearRegression import LinearRegression

model = LinearRegression(file_path='data.txt', theta=[0, 0], num_iter=15000, learning_rate=0.01, verbose=0)

model.train()

model.plot_line()

model.plot_cost()

model.compare_to_Sklearn()