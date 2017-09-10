# -*- coding: utf-8 -*-
import numpy as np
from Perceptron_original import Perceptron

data = np.genfromtxt('./iris2.txt')
X = data[:,0:4]
y = data[:,4]

model = Perceptron()
model.fit(X,y)
print(model.errors_)
print(model.w_)
y_predic = model.predict(X)