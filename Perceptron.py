import numpy as np
from matplotlib import pyplot as plt


class Perceptron(object):
    """Perceptron classifer.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter :  int
        Passes over the training dataset.

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in ever epoch.

    """
    def __init__(self, eta=0.01,  n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        :param X: {array-like}, shape = [n_samples, n_features]
                  Training vectors, where n_samples
                  is the number of samples and
                  n_features is the number of features.
        :param y: array-like, shape = [n_samples]
                  Target values.
        :return:  self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            #errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                #if update!=0:
                print ('xi', xi, self.predict(xi), 'target', target, update)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                

            #self.errors_.append(errors)    
            self.errors_.append(np.sum(np.not_equal(y,self.predict(X))))
            self.getboundary()

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def getboundary(self):
        self.boundx = np.arange(4, 7.5, 0.05)
        self.boundy = (-self.w_[0] - self.w_[1] * self.boundx) / self.w_[2]
        plt.scatter(X[:50,0], X[:50,1], c = 'r')
        plt.scatter(X[50:100,0], X[50:100,1], c = 'b')
        plt.scatter(self.boundx, self.boundy, c = 'black')
        plt.show()
        print(self.w_)
                
        
def getboundary():
    return 0

print(getboundary())


        
data = np.genfromtxt('./iris.txt')
X = data[:,0:2]
y = data[:,2]

model = Perceptron(eta = 0.01, n_iter = 3)

model.fit(X, y)
predict = model.predict(X)

print(model.errors_)
print(model.w_)

'''
boundx = np.arange(4, 7.5, 0.1)
boundy = (-model.w_[0] - model.w_[1]*boundx) / model.w_[2]





plt.scatter(X[:50,0], X[:50,1], c = 'r')
plt.scatter(X[50:100,0], X[50:100,1], c = 'b')
plt.scatter(boundx, boundy, c = 'black')
plt.show()
'''