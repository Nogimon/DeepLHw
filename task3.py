import numpy as np
from matplotlib import pyplot as plt
from Perceptron_multi import Perceptron_multi

data = np.genfromtxt('./iris.txt')
X = data[:,0:2]
y = data[:,2]

model = Perceptron_multi(0.01, 10)
#model.fit(X,y)
#model.fit_batch(X, y)
print("Now start with online")
model.fit(X, y)
print("The errors are", model.errors_)
print("\nNow start with fullbatch")
model.fit_batch(X, y)
print("The errors are", model.errors_)
print("\nNow start with minibatch")
model.fit_minibatch(X, y)
print("The errors are", model.errors_)
y_predic = model.predict(X)

print("\nNow try with different learning rate")
for rate in {0.1, 0.01, 0.001}:
	print("Now uses rate", rate)
	model = Perceptron_multi(rate, 10)
	model.fit_batch(X, y)
	print("The errors are", model.errors_)

print("\nNow try with different initialization weights")
for weights in [[1, -1, 1], [5, -5, 5], [500, -500, 500]]:
	print("Now uses weights", weights)
	model = Perceptron_multi(0.01, 20, weights)
	model.fit_batch(X, y)
	print("The errors are", model.errors_)