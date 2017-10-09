import numpy as np
import keras
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
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
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.reshape(x_train, (50000, 32*32*3))
x_test = np.reshape(x_test, (10000, 32*32*3))

y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

model = Sequential()

model.add(Dense(units = 30, input_dim = 32*32*3))#input_shape = (28,28)))
model.add(Activation('relu'))
model.add(Dense(units = 10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-5), metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizer.SGD(lr = 0.01, momentum = 0.9, nesterov = True), metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 30, batch_size = 10)

pre = model.predict(x_test)
pre = [np.argmax(x) for x in pre]
y = [np.argmax(y) for y in y_test]

acc = 0
for i in range(len(pre)):
    if pre[i] == y[i]:
        acc+=1

score = model.evaluate(x_test, y_test, batch_size = 100)

#convolutional network
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (32, 32, 3)))#input_shape = (28,28)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-5), metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 30, batch_size = 10)

pre = model.predict(x_test)
pre = [np.argmax(x) for x in pre]
y = [np.argmax(y) for y in y_test]

acc = 0
for i in range(len(pre)):
    if pre[i] == y[i]:
        acc+=1

score = model.evaluate(x_test, y_test, batch_size = 100)