# -*- coding: utf-8 -*-
import numpy as np
from Perceptron_original import Perceptron
from Perceptron_logistic import Perceptron_logistic

print("now start to read iris2")
data = np.genfromtxt('./iris2.txt')
X = data[:,0:4]
y = data[:,4]

model = Perceptron()
model.fit(X,y)
print(model.errors_)
print(model.w_)
y_predic = model.predict(X)



print("now start to do 10 fold validation")
split_number = 10
#X_vali = np.split(X, split_number)
#y_vali = np.split(X, split_number)

for i in range(0, 10):
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
    print(i, 'th round, the final error is', error)

    
print("now start to use logistic neuron")
model = Perceptron_logistic(eta = 100, n_iter = 50)

y[y == -1] = 0
model.fit_batch(X, y)
print(model.errors_)
print(model.w_)