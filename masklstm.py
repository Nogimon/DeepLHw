from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
import numpy as np
import pylab as plt
from PIL import Image
import cv2
from glob import glob

def loaddata(directory):
	data = []

	file = glob(directory + "*mask.png")
	for i in file:
		img = np.asarray(Image.open(i))
		resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)
		data.append(resized)

	data = np.asarray(data)
	return data

def get_lstm():
	inputs = Input((None, 128, 128, 1))
	lstm1 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
	norm1 = BatchNormalization()(lstm1)
	lstm2 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(norm1)
	norm2 = BatchNormalization()(lstm2)
	lstm3 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(norm2)
	norm3 = BatchNormalization()(lstm3)
	lstm4 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(norm3)
	norm4 = BatchNormalization()(lstm4)
	conv1 = Conv3D(filters=1, kernel_size=(3, 3, 3),
	               activation='sigmoid',
	               padding='same', data_format='channels_last')(norm4)
	model = Model(inputs = [inputs], outputs = [conv1])
	return model

if __name__ == "__main__":
	directory = "/media/zlab-1/Data/Lian/course/DeepLHw/data/"
	data = loaddata(directory)
	data = data.reshape(34, 10, 128, 128)
	data=data[...,np.newaxis]

	model = get_lstm()
	model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

	train = data[:-1]
	test = data[1:]

	model.fit(train, test, batch_size = 10, epochs = 10, validation_split = 0.05)

	'''
	noisy_movies, shifted_movies = generate_movies(n_samples=1200)
	model.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=10, epochs=10, validation_split=0.05)
	'''
