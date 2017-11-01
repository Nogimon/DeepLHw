import numpy as np
import pickle as cPickle
import gzip
import random
from network import Network
from mnist_loader import load_data, load_data_wrapper, vectorized_result

#Task 1.2
network = Network([784, 30, 10])
training_data, validation_data, test_data = load_data_wrapper()


print("\ntask1.2:\nnow start to train\n", file = open("result.txt", "a"))
network.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#Task 1.3
print("\ntask1.3\nnow start to train with 0 hidden layers\n", file = open("result.txt", "a"))
network1 = Network([784, 10])
network1.SGD(training_data, 30, 10, 3.0, test_data = test_data)

print("\nnow start to train with learning rate = 30", file = open("result.txt", "a"))
network.SGD(training_data, 30, 10, 30, test_data = test_data)

print("\nnow start to train with learning rate = 0.01", file = open("result.txt", "a"))
network.SGD(training_data, 30, 10, 0.01, test_data = test_data)

#Task 1.4
#See README for analysis result

#Task 1.5
print("\nnow start to train with softmax layer and cross entropy cost", file = open("result.txt", "a"))
network.SGD_softmax(training_data, 30, 10, 3.0, activation_function = "sigmoid", test_data = test_data)

#Task 1.6
print("\nnow start to train with ReLU activation function", file = open("result.txt", "a"))
network.SGD_softmax(training_data, 30, 10, 0.15, activation_function = "ReLU", test_data = test_data)

#Task 1.7
print("\nnow start to train with Leaky ReLU activation function", file = open("result.txt", "a"))
network.SGD_softmax(training_data, 30, 10, 0.15, activation_function = "LeakyReLU", test_data = test_data)
