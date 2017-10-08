import numpy as np
import keras
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
'''
x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size = (1000, 1))
y_train = keras.utils.to_categorical(y_train, num_classes = 10)

model = Sequential()

model.add(Dense(units = 30, input_dim = 100))#input_shape = (28,28)))
model.add(Activation('relu'))
model.add(Dense(units = 10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-5), metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizer.SGD(lr = 0.01, momentum = 0.9, nesterov = True), metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5, batch_size = 10)
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))

y_train = keras.utils.to_categorical(y_train, num_classes = 10)

model = Sequential()

model.add(Dense(units = 30, input_dim = 784))#input_shape = (28,28)))
model.add(Activation('relu'))
model.add(Dense(units = 10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-5), metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizer.SGD(lr = 0.01, momentum = 0.9, nesterov = True), metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5, batch_size = 100)