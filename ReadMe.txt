
****Task 1****
1.1
In the original "Perceptron.py", it uses Online updating strategy

1.2
Stochastic gradient descent is apart from gradient descent. In gradient descent the gradient is calculated over all the data, but stochastic gradient descent only uses single or a few training examples to calculate the gradient and then do the update.
Since often the dataset has a very large quantity. In these cases using stochastic gradient descent can be highly efficient.


****Task 2****
(Note: in all my files I changed error_ to display the # of wrong predictons)
2.1
I have used Python 3.5 from Anaconda and Spyder as my IDE.

2.2



****Task 3****




****Task 4****
All task 4 problems are done in task4.py. By simply run
$ python task4.py
or run task4.py in any IDE can generate the answers

4.1
As shown in the output from the python code.
Error rate(# of misclassified cases) change and final weights are printed out 

4.2
Cross validation is to separate the training data into parts. For each iteration do testing to the part while do training to the others, and then do the iteration through all parts. Thus every part has been act as testing data, to show how the model is behaved.

4.3
As shown in the output from the python code.
k = 10 is used in my code. And k can be changed in the code by changing the variable 'split_number'
Error rate of each fold and final average weight vector is reported.

4.4
new module is written in Perceptron_logistic.py and is imported in task4.py
For the Perceptron_logistic, the predict function and error function is re-written.
The new predict function now returns

return (1 / (1 + np.exp(-self.net_input(X))))

And since a logistic function is used, the previous error function which counts the # of misclassified component cannot be used. Therefore a new error function is written as

self.errors_.append(np.sum(0.5 * np.power(y - self.predict(X), 2)))

which gives a simple square of the diff from target and prediction.
The update mechanism is also updated in fit_batch function.
The error rate is printed out by the program. In order to get obvious improvement 30 iterations are used.
In the end the weights are the training speed and accuracy of the logistic neuron cannot compare to the ones of linear neuron.