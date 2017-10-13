import numpy as np
import pickle as cPickle
import gzip
import random
from network_2 import Network
from network_3 import Network3
from mnist_loader import load_data, load_data_wrapper, vectorized_result

def mean_data(data):
    data_mean = []
    for i in range(len(data)):
        datax = list(data[i][0])
        datay = data[i][1]
        datax -= np.mean(datax, axis = 0)
        datax /= np.std(datax, axis = 0)
        datan = (datax, datay)
        data_mean.append(datan)
    return data_mean


#Task 2.1
network = Network([784, 30, 10])
training_data, validation_data, test_data = load_data_wrapper()

print("\ntask2.1:\nnow start to train\n", file = open("result_task2.txt", "a"))
network.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#Task 2.2
train_mean = mean_data(training_data)
test_mean = mean_data(test_data)
print("\ntask2.2:\nnow start to train with prepocessed data\n", file = open("result_task2.txt", "a"))
network.SGD(train_mean, 30, 10, 3.0, test_data=test_mean)


#Task 2.3
network2 = Network([784, 30, 10])
print("\ntask2.2:\nnow start to train with momentum data\n", file = open("result_task2.txt", "a"))
network2.SGD(training_data, 30, 10, 3.0, update = "momentum", test_data=test_data)

print("\ntask2.3:\nnow start to train with Nestorov data\n", file = open("result_task2.txt", "a"))
network2.SGD(training_data, 30, 10, 3.0, update = "Nestorov", test_data=test_data)

#Task 2.4
print("\ntask2.4:\nnow start to train with dropout\n", file = open("result_task2.txt", "a"))
network3 = Network3([784, 30, 10])
network3.SGD(training_data, 30, 10, 3.0, dropout = "true", test_data=test_data)

#Task 2.5
print("\ntask2.5:\nnow start with Model Averaging\n", file = open("result_task2.txt", "a"))
sum1 = 0
for (x, y) in test_data:
	t1 = network.feedforward(x)
	t2 = network2.feedforward(x)
	t3 = network3.feedforward(x)
	t = t1 + t2 +t3
	t = np.argmax(t)
	if t == y:
		sum1 += 1
print ("averaged result: {0} / {1}".format(sum1, len(test_data)))

#Task 2.6
print("\ntask2.6:\nnow start with Adaptive learning rate\n", file = open("result_task2.txt", "a"))
network3.SGD(training_data, 30, 10, 3.0, adapteta = "true", test_data=test_data)