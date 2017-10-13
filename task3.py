import numpy as np
import keras
from keras.datasets import cifar10, mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from network import Network

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1
    
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#Old methods
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



#Convolutional network
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (32, 32, 3)))#input_shape = (28,28)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
'''
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(2048))
model.add(Dropout(0.5))
'''
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-5), metrics = ['accuracy'])

#model = load_model('./weights1.h5')

model_checkpoint = ModelCheckpoint('./weights1.h5'.format(), monitor='val_loss', save_best_only=True)

model.fit(x_train, y_train, epochs = 80, batch_size = 10, callbacks=[model_checkpoint], validation_data=(x_test, y_test))

pre = model.predict(x_test)
pre = [np.argmax(x) for x in pre]
y = [np.argmax(y) for y in y_test]

acc = 0
for i in range(len(pre)):
    if pre[i] == y[i]:
        acc+=1

score = model.evaluate(x_test, y_test, batch_size = 100)

print (acc)