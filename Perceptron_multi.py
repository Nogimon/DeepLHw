import numpy as np
from matplotlib import pyplot as plt


class Perceptron_multi(object):
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
                self.w_[1:] += update * xi
                self.w_[0] += update
                #errors += int(update != 0.0)
                
            self.errors_.append(np.sum(np.not_equal(y,self.predict(X))))
            print(self.w_)
            self.getboundary()
        return self
        
    def fit_batch(self, X, y):
        #self.w_ = [-4, -3, 19]
        self.w_ = [0, -1, 1]
        self.errors_ = []
        self.eta /= 100 #For full batch the eta needs to be set smaller
        for _ in range(self.n_iter):
            update = 0
            for xi, target in zip(X, y):
                update += self.eta * (target - self.predict(xi))
            [x1, x2] = np.sum(X, axis = 0)
            update /= len(X)
            print(self.w_, x1, x2, update)
            self.w_[1] += update * x1
            self.w_[2] += update * x2
            self.w_[0] += update
            self.errors_.append(np.sum(np.not_equal(y,self.predict(X))))
            print("new weight", self.w_)
            self.getboundary()    
        
        return self
        
    def fit_minibatch(self, X, y):
        #self.w_ = [-4, -3, 19]
        self.w_ = [0, -1, 1]
        self.errors_ = []
        self.eta /= 1
        for _ in range(self.n_iter):
            update = 0
            i = 0
            start = 0
            for xi, target in zip(X, y):
                update += self.eta * (target - self.predict(xi))
                i += 1
                if ((i % 20) == 0):
                    [x1, x2] = np.sum(X[start:i], axis = 0)
                    update /= 20
                    self.w_[1] += update * x1
                    self.w_[2] += update * x2
                    self.w_[0] += update
                    print(self.w_, x1, x2, update, i)
                    update = 0
                    start = i
                    print(self.w_, x1, x2, update, i)
            
            self.errors_.append(np.sum(np.not_equal(y,self.predict(X))))
            print("new weight", self.w_)
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
        #print(self.w_)

        
data = np.genfromtxt('./iris.txt')
X = data[:,0:2]
y = data[:,2]

model = Perceptron_multi(0.01, 10)
#model.fit(X,y)
#model.fit_batch(X, y)
model.fit_minibatch(X, y)
print(model.errors_)
y_predic = model.predict(X)