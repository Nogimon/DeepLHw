# -*- coding: utf-8 -*-
import numpy as np
from Perceptron_original import Perceptron
from Perceptron_logistic import Perceptron_logistic

print("Now start with 4.1")
print("now start to read iris2")
data = np.genfromtxt('./iris2.txt')
X = data[:,0:4]
y = data[:,4]

model = Perceptron()
model.fit(X,y)
print("the errors through training are", model.errors_)
print("the final weights are", model.w_)
y_predic = model.predict(X)


print("\nNow start to do 4.3")
split_number = 10
print("now start to do ", split_number, " fold validation")
average_weight = np.zeros(5)

for i in range(0, split_number):
    part = int(len(X)/split_number)
    start = (i * part)
    end = (i+1) * part

    X_test = X[start:end,:]
    X_train = np.vstack((X[:start,:], X[end:,:]))
    y_test = y[start:end]
    y_train = np.append(y[:start], y[end:])
    
    model = Perceptron()
    model.fit(X_train, y_train)
    print(i, "th round, the error within is ", model.errors_)
    error = np.sum(np.not_equal(y_test, model.predict(X_test)))
    print(i, 'th round, the final weight is', model.w_)
    average_weight = (average_weight + model.w_) / 2

print("The final average weights are", average_weight)

print("\nNow start to do 4.4")   
print("now start to use logistic neuron")
model = Perceptron_logistic(eta = 0.01, n_iter = 30)

y[y == -1] = 0
model.fit_batch(X, y)
print("the errors are", model.errors_)
print("the final weights are", model.w_)
