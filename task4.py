import numpy as np
import keras
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam


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

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
'''
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

model_checkpoint = ModelCheckpoint('./weights_task4_1.h5'.format(), monitor='val_loss', save_best_only=True)

model.fit(x_train, y_train, epochs = 80, batch_size = 10, callbacks=[model_checkpoint], validation_data=(x_test, y_test))

pre = model.predict(x_test)
pre = [np.argmax(x) for x in pre]
y = [np.argmax(y) for y in y_test]

acc = 0
for i in range(len(pre)):
    if pre[i] == y[i]:
        acc+=1

score = model.evaluate(x_test, y_test, batch_size = 100)


