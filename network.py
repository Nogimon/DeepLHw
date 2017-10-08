"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation. 
"""

#### Libraries
# Standard library
import random
import math

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def feedforward_softmax(self, activation, activation_function):
        count = 1
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            if count < self.num_layers-1:
                if activation_function == "sigmoid":
                    activation = sigmoid(z)
                elif activation_function == "LeakyReLU":
                    activation = LeakyReLU(z)
                else:
                    activation = ReLU(z)               
            else:
                #Switch the last layer with softmax one
                activation = softmax(z)

            count += 1
        return activation

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test), file = open("result.txt", "a"))
            else:
                print ("Epoch {0} complete".format(j))
                print ("Epoch {0} complete".format(j), file = open("result.txt", "a"))
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def SGD_softmax(self, training_data, epochs, mini_batch_size, eta, activation_function, test_data=None):

        if activation_function == "ReLU" or activation_function == "LeakyReLU":
            self.biases = [0.01 + 0.01 * np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = [0.01 * np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
            #self.weights = [np.random.randn(n) * math.sqrt(2.0/n) + 0.01 for n in self.sizes[1:]]
            #self.biases = [np.random.randn(n) * math.sqrt(2.0/n) + 0.01 for n in self.sizes[1:]]
            #for bias in self.biases:
            #    bias.fill(0.01)

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_softmax(mini_batch, eta, activation_function)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate_softmax(test_data, activation_function), n_test))
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate_softmax(test_data, activation_function), n_test), file = open("result.txt", "a"))
                """
                if self.evaluate(test_data) > 6500:
                    eta = 0.0015
                elif self.evaluate(test_data) > 7500:
                    eta = 0.00015
                """
            else:
                print ("Epoch {0} complete".format(j))
                print ("Epoch {0} complete".format(j), file = open("result.txt", "a"))
    def update_mini_batch_softmax(self, mini_batch, eta, activation_function):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop_softmax(x, y, activation_function)
            #if delta_nabla_b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    def backprop_softmax(self, x, y, activation_function):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        count = 1
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            if count < self.num_layers-1:
                if activation_function == "sigmoid":
                    activation = sigmoid(z)
                elif activation_function == "LeakyReLU":
                    activation = LeakyReLU(z)
                else:
                    activation = ReLU(z)               
            else:
                #Switch the last layer with softmax one
                activation = softmax(z)
            activations.append(activation)
            count += 1
        # backward pass
        #switch the last delta with the cross entropy delta for softmax
        delta = self.cost_derivative_crossentropy(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        if activation_function == "ReLU":
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = ReLU_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        elif activation_function == "LeakyReLU":
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = LeakyReLU_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        else:
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_softmax(self, test_data, activation_function):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward_softmax(x, activation_function)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def cost_derivative_crossentropy(self, output_activations, y):
        return (output_activations - y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

#input an array, return a softmax array
def softmax(z):
    summation = np.sum(np.exp(z))
    return (np.exp(z) / summation)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def ReLU(z): 
    return np.maximum(0.0, z)

def ReLU_prime(z):
    return 1 * (z > 0)

def LeakyReLU(z):
    return np.maximum(0.01 * z, z)
    #return z * (z > 0) + 0.01 * z * (z < 0)
    
def LeakyReLU_prime(z):
    return 1 * (z > 0)  - 0.01 * (z < 0)
