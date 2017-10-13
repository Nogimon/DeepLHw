
from network import Network
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1
    
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
#Start to read cifar data
data = unpickle('./cifar-10-batches-py/data_batch_1')
training_inputs0 = data[b'data']
training_results0 = data[b'labels']
#Process the data into tuples to reach the requirements for network.py
training_inputs1 = training_inputs0[0:9000]
test_inputs1 = training_inputs0[9000:]
training_results1 = training_results0[0:9000]
test_result1 = training_results0[9000:]

training_inputs2 = [np.reshape(x, (3072, 1)) for x in training_inputs1]
training_results2 = [vectorized_result(y) for y in training_results1]
training_data1 = list(zip(training_inputs2, training_results2))

test_inputs = [np.reshape(x, (3072, 1)) for x in test_inputs1]
test_data = list(zip(test_inputs,test_result1))
#build and train the network
network = Network([3072, 35, 10])
network.SGD_softmax(training_data1, 30, 10, 0.15, activation_function = "LeakyReLU", test_data = test_data)