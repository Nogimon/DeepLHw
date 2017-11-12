
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from matplotlib.patches import Polygon
import cv2

def generateData():
	#set parameters
	dire = '/media/zlab-1/Data/Lian/course/DeepLHw/coco/'
	dataDir=dire
	dataType='val2014'
	annFile='{}/annotations/instances_{}.json'.format(dire,dataType)

	#load coco api and coco categories
	coco=COCO(annFile)

	cats = coco.loadCats(coco.getCatIds())
	nms=[cat['name'] for cat in cats]
	print('COCO categories: \n{}\n'.format(' '.join(nms)))

	nms = set([cat['supercategory'] for cat in cats])
	print('COCO supercategories: \n{}'.format(' '.join(nms)))

	#load data
	catIds = coco.getCatIds(catNms = ['person'])
	imgIds = coco.getImgIds(catIds = catIds)
	img = coco.loadImgs(imgIds)

	#annIds = coco.getAnnIds(catIds = catIds)
	#anns = coco.loadAnns(imgIds)

	train = []
	y = []
	for i in range(len(img)):
		I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img[i]['file_name']))
		I2 = cv2.resize(I, (128, 128), cv2.INTER_LINEAR)

		annIds = coco.getAnnIds(imgIds=img[i]['id'], catIds=catIds, iscrowd=None)
		anns = coco.loadAnns(annIds)
		mask = coco.annToMask(anns[0])
		mask2 = cv2.resize(mask, (128, 128), cv2.INTER_LINEAR)

		train.append(I2)
		y.append(mask2)
	

	return X, y


def get_model(channelsize):
	inputs = Input((128,128, 1))
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
	conv10 = Conv2D(channelsize, (1, 1), activation = 'sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])
	return model


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def generateData():
    



if __name__ == '__main__':

	channelsize = 5
    train, y, X_validate, y_validate = generateData()


    #Set the model
    K.set_image_data_format('channels_last') 
    smooth=1.
    model=get_model(channelsize)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model_checkpoint = ModelCheckpoint(directory+'/weights.h5', monitor='val_loss', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
    
    

    #Train
    model.fit(train, y, batch_size=32, epochs=150, verbose=1, shuffle=True, callbacks=[model_checkpoint, earlystop],validation_data=(X_test, y_test))

    #Test
    #a=model.predict(X_test, batch_size=32, verbose=2)
