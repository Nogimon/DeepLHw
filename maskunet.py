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
	y = []
	shift = 10

	file0 = glob(directory + "*[0-9].png")
	for i in file0:
		img = np.asarray(Image.open(i))
		resized = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
		train.append(resized)

		#augmentation
		augmented = transform.rotate(resized, 90)
		train.append(augmented)
		augmented = transform.rotate(resized, 180)
		train.append(augmented)
		augmented = transform.rotate(resized, 270)
		train.append(augmented)
		augmented = np.roll(resized, shift, axis = 1)
		augmented[:,0:shift] = 0
		train.append(augmented)
		augmented = np.roll(resized, shift, axis = 0)
		augmented[0:shift,:] = 0
		train.append(augmented)
		augmented = np.roll(resized, -shift, axis = 1)
		augmented[:,-shift:] = 0
		train.append(augmented)
		augmented = np.roll(resized, -shift, axis = 0)
		augmented[-shift:,:] = 0
		train.append(augmented)
	
	file1 = glob(directory + "*mask.png")
	for i in file1:
		img = np.asarray(Image.open(i))
		resized = cv2.resize(img, (size, size), cv2.INTER_LINEAR)
		y.append(resized)

		#augmentation
		augmented = transform.rotate(resized, 90)
		y.append(augmented)
		augmented = transform.rotate(resized, 180)
		y.append(augmented)
		augmented = transform.rotate(resized, 270)
		y.append(augmented)
		augmented = np.roll(resized, shift, axis = 1)
		augmented[:,0:shift] = 0
		y.append(augmented)
		augmented = np.roll(resized, shift, axis = 0)
		augmented[0:shift,:] = 0
		y.append(augmented)
		augmented = np.roll(resized, -shift, axis = 1)
		augmented[:,-shift:] = 0
		y.append(augmented)
		augmented = np.roll(resized, -shift, axis = 0)
		augmented[-shift:,:] = 0
		y.append(augmented)

	train = np.asarray(train)
	y = np.asarray(y)

	return (train, y)

def get_unet():
	inputs = Input(shape = (128, 128, 1))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	#conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
	conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])
	
	return model

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

def plotresult(a, groundtruth, original):

	for i in range(len(a)):
		c=a[i]*255
		c=c.astype('uint8')
		b=groundtruth[i]*255
		bb=cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
		cc=cv2.cvtColor(c,cv2.COLOR_GRAY2RGB)
		bbb=np.where(bb>0)
		ccc=np.where(cc>0)

		cc[ccc[0:2]]=[255, 0, 0]
		bb[bbb[0:2]]=[0, 0, 255]   

		dst = cv2.addWeighted(cc,0.5,bb,0.5,0)    
		cv2.imwrite('./result/{}.jpg'.format(format(i,'05')),dst)
		cv2.imwrite('./result/{}_original.jpg'.format(format(i,'05')), original[i])



if __name__ == "__main__":
	size = 128
	directory = "./data/"
	train, y = loaddata(directory, size)

	train = train[..., np.newaxis]
	y = y[..., np.newaxis]

	#Set the model
	K.set_image_data_format('channels_last') 
	smooth=1.
	model=get_unet()
	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
	model_checkpoint = ModelCheckpoint('./weights.h5', monitor='val_loss', save_best_only=True)
	earlystop = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

	x_test = train[:100]
	y_test = y[:100]

	#Train
	model.fit(train[100:], y[100:], batch_size=32, epochs=30, verbose=1, shuffle=True, callbacks=[model_checkpoint, earlystop],validation_data=(x_test, y_test))

	#Test
	predict = model.predict(train, batch_size=32, verbose=2)

	plotresult(predict, y, train)

