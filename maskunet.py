from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage import transform

def loaddata(directory, size):
	train = []
	gt = []

	file = glob(directory + "*mask.png")
	for i in file:
		img = np.asarray(Image.open(i))
		resized = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
		data.append(resized)

	data = np.asarray(data)
	return data