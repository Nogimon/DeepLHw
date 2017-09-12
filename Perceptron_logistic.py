import numpy as np
class Perceptron_logistic(object):
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
            #self.errors_.append(errors)
            self.errors_.append(np.sum(np.not_equal(y,self.predict(X))))
        return self

    def fit_batch(self, X, y):
        #self.w_ = [-4, -3, 19]
        #self.w_ = np.zeros(1 + X.shape[1])
        self.w_ = [-1, 1, 1, 1, 1]
        self.errors_ = []
        #self.eta /= 100 #For full batch the eta needs to be set smaller
        for _ in range(self.n_iter):
            update = 0
            update0 = 0
            for xi, target in zip(X, y):
                #update += self.eta * (target - self.predict(xi))
                y1 = self.predict(xi)
                update += -self.eta * (xi * y1 * (1 - y1) * (target - y1))
                update0 += -self.eta * (y1 * (1 - y1) * (target - y1))
            #[x1, x2] = np.sum(X, axis = 0)
            #update /= len(X)
            print('the weights are', self.w_)
            print('the update are', update)
            self.w_[0] += update0
            self.w_[1] += update[0]
            self.w_[2] += update[1]
            self.w_[3] += update[2]
            self.w_[4] += update[3]
            self.errors_.append(np.sum(0.5 * np.power(y - self.predict(X), 2)))
            print(self.predict(X))
            print("new weight", self.w_)
            #self.getboundary()    
        
        return self
        
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        #return np.where(self.net_input(X) >= 0.0, 1, -1)
        return (1 / (1 + np.exp(self.net_input(X))))

        

