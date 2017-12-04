from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
from PIL import Image

def loaddata(directory):
	train = []
	test = []

	file = glob(directory + ".mask.png")
	for i in file:
		img = np.asarray(Image.open(i))



if __name__ == "__main__":
	directory = "./data"
	train, test = loaddata(directory)
	