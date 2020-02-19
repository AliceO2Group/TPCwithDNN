import matplotlib.pyplot as plt # matplotlib module to make the nice plots
import numpy as np  # numpy module
import pandas as pd # pandas dataframe
import keras
from os.path import isfile
from keras.models import Sequential, Model
from keras.layers import Input, concatenate, Flatten, UpSampling3D, AveragePooling3D, ZeroPadding3D
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
import gc

from keras.models import model_from_json
import time
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import os
import h5py
import itertools
from SymmetricPadding3D import SymmetricPadding3D
# Reset Keras Session
def ResetKeras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))



# read TChain or TFile save each in separated dataframe
def GetLocalCurrentNumpyArray(fnames):
	vecFluctuationSC = []
	vecFluctuationDistR = []
	step = 0
	for fname in fnames:
		if isfile(fname + '.h5'):
			print("found h5 file")
			hf = h5py.File(fname + '.h5', 'r')
			dataSet = np.array(hf["meanCurrent"][:])
			print(dataSet[0][:])
			print("len : " + str(len(dataSet[0][:])))
			if (len(vecFluctuationSC) == 0):
				vecFluctuationSC = dataSet[0][:]
				vecFluctuationDistR = dataSet[1][:]
			else:
				vecFluctuationSC = np.concatenate([vecFluctuationSC[:],dataSet[0][:]])
				vecFluctuationDistR = np.concatenate([vecFluctuationDistR[:],dataSet[1][:]])
		else:	
			if not isfile(fname):
				print('file is not found:' + fname)
				return 0
		
	return [np.array(vecFluctuationSC),np.array(vecFluctuationDistR)]			


# Getmodel from disk
def GetModelFromDisk(phiSlice,rRow,zColumn,beginSize,dropOut=0.0,poolingType = 0):
	modelDir = os.environ['MODELDIR']

	print("load model from file")
	start = time.time() 
	json_file = open(modelDir + "modelLocalCurrent{}-{}-{}-{}-{}-{}.json".format(phiSlice,rRow,zColumn,beginSize,poolingType,dropOut), "r") 
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json,{'SymmetricPadding3D' : SymmetricPadding3D})
	loaded_model.summary()
	loaded_model.load_weights(modelDir + "modelLocalCurrent{}-{}-{}-{}-{}-{}.h5".format(phiSlice,rRow,zColumn,beginSize,poolingType,dropOut)) 
	end = time.time()
	load_time = end - start
	return [load_time,loaded_model]


# generate model of CNN
# model the input is (16,16,16) the output is (16,16,16)
def GetModel(modelId,phiSlice,rRow,zColumn,size,dropout=0.0,poolingtype = 0, batch_normalization=True):
	if modelId == 0:
		myinput = Input((phiSlice,rRow,zColumn,1)) 
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1)
		conv1 = Dropout(dropout)(conv1)	
		if poolingtype == 0:
			pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
		else:
			pool1 = AveragePooling3D(pool_size=(2,2,2))(conv1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2)
		conv2 = Dropout(dropout)(conv2)	
		if poolingtype == 0:
			pool3 = MaxPooling3D(pool_size=(2,2,2))(conv2)
		else:
			pool3 = AveragePooling3D(pool_size=(2,2,2))(conv2)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3)
		conv3 = Dropout(dropout)(conv3)	
		if poolingtype == 0:
			pool4 = MaxPooling3D(pool_size=(2,2,2))(conv3)
		else:
			pool4 = AveragePooling3D(pool_size=(2,2,2))(conv3)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4)
		conv4 = Dropout(dropout)(conv4)	
		if poolingtype == 0:
			pool5 = MaxPooling3D(pool_size=(2,2,2))(conv4)
		else:
			pool5 = AveragePooling3D(pool_size=(2,2,2))(conv4)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5)
		conv5 = Dropout(dropout)(conv5)	
		up6   = UpSampling3D(size=(2,2,2))(conv5)
		up6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(up6)
		merge6 = concatenate([conv4,up6]) 
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6)
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6)
		up7   = UpSampling3D(size=(2,2,2))(conv6)
		up7 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(up7)
		merge7 = concatenate([conv3,up7]) 
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7)
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8)
		up7   = UpSampling3D(size=(2,2,2))(conv8)
		up7   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)))(up7)
		merge8 = concatenate([conv2,up7]) 
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8)
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9)
		up8   = UpSampling3D(size=(2,2,2))(conv9)
		up8   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)))(up8)
		merge9 = concatenate([conv1,up8]) 
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9)
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		conv10 = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10) 
		model = Model(inputs = myinput, outputs = conv10) 
	if modelId == 1:
		myinput = Input((phiSlice,rRow,zColumn,1)) 
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1)
		conv1 = Dropout(dropout)(conv1)	
		if poolingtype == 0:
			pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
		else:
			pool1 = AveragePooling3D(pool_size=(2,2,2))(conv1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2)
		conv2 = Dropout(dropout)(conv2)	
		if poolingtype == 0:
			pool3 = MaxPooling3D(pool_size=(2,2,2))(conv2)
		else:
			pool3 = AveragePooling3D(pool_size=(2,2,2))(conv2)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3)
		conv3 = Dropout(dropout)(conv3)	
		if poolingtype == 0:
			pool4 = MaxPooling3D(pool_size=(2,2,2))(conv3)
		else:
			pool4 = AveragePooling3D(pool_size=(2,2,2))(conv3)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4)
		conv4 = Dropout(dropout)(conv4)	
		if poolingtype == 0:
			pool5 = MaxPooling3D(pool_size=(2,2,2))(conv4)
		else:
			pool5 = AveragePooling3D(pool_size=(2,2,2))(conv4)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5)
		conv5 = Dropout(dropout)(conv5)	
		up6   = UpSampling3D(size=(2,2,2))(conv5)
		up6   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6)
		merge6 = concatenate([conv4,up6]) 
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6)
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6)
		up7   = UpSampling3D(size=(2,2,2))(conv6)
		merge7 = concatenate([conv3,up7]) 
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7)
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8)
		up8   = UpSampling3D(size=(2,2,2))(conv8)
		up8   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up8)
		merge8 = concatenate([conv2,up8]) 
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8)
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9)
		up9   = UpSampling3D(size=(2,2,2))(conv9)
		up9   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="SYMMETRIC")(up9)
		merge9 = concatenate([conv1,up9]) 
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9)
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		conv10 = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10) 
		model = Model(inputs = myinput, outputs = conv10) 
	if modelId == 2: # 33-33-180 
		myinput = Input((phiSlice,rRow,zColumn,1)) 
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1)
		conv1 = Dropout(dropout)(conv1)	
		if poolingtype == 0:
			pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
		else:
			pool1 = AveragePooling3D(pool_size=(2,2,2))(conv1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2)
		conv2 = Dropout(dropout)(conv2)	
		if poolingtype == 0:
			pool3 = MaxPooling3D(pool_size=(2,2,2))(conv2)
		else:
			pool3 = AveragePooling3D(pool_size=(2,2,2))(conv2)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3)
		conv3 = Dropout(dropout)(conv3)	
		if poolingtype == 0:
			pool4 = MaxPooling3D(pool_size=(2,2,2))(conv3)
		else:
			pool4 = AveragePooling3D(pool_size=(2,2,2))(conv3)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4)
		conv4 = Dropout(dropout)(conv4)	
		if poolingtype == 0:
			pool5 = MaxPooling3D(pool_size=(2,2,2))(conv4)
		else:
			pool5 = AveragePooling3D(pool_size=(2,2,2))(conv4)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5)
		conv5 = Dropout(dropout)(conv5)
		if poolingtype == 0:
			pool5a = MaxPooling3D(pool_size=(2,2,2))(conv5)
		else:
			pool5a = AveragePooling3D(pool_size=(2,2,2))(conv5)
		conv5a = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5a)
		conv5a = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5a)
		conv5a = Dropout(dropout)(conv5a)
	
		up6a   = UpSampling3D(size=(2,2,2))(conv5a)
		#up6a   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)))(up6a)
		up6a   = SymmetricPadding3D(padding=[[1,0],[0,0],[0,0]],mode="SYMMETRIC")(up6a)

		merge6a = concatenate([conv5,up6a])
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6a)
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6a)


		up6   = UpSampling3D(size=(2,2,2))(conv6a)
		#up6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(up6)
		merge6 = concatenate([conv4,up6])
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6)
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6)
		up7   = UpSampling3D(size=(2,2,2))(conv6)
		up7   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up7)
		#up7 = Conv3D(4*size,3, activation="relu", padding="valid", kernel_initializer ="normal")(up7)
		merge7 = concatenate([conv3,up7]) 
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7)
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8)
		up8   = UpSampling3D(size=(2,2,2))(conv8)
		#up8 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(up8)
		merge8 = concatenate([conv2,up8]) 
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8)
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9)
		up9   = UpSampling3D(size=(2,2,2))(conv9)
		up9   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="REFLECT")(up9)
		#up9 = Conv3D(size,3, activation="relu", padding="valid", kernel_initializer ="normal")(up9)	
		merge9 = concatenate([conv1,up9]) 
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9)
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		#conv10 = Conv3D(size,2, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		conv10 = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10) 
		model = Model(inputs = myinput, outputs = conv10) 
	if modelId == 3: # 65-65-180 
		myinput = Input((phiSlice,rRow,zColumn,1)) 
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1)
		conv1 = Dropout(dropout)(conv1)	
		if poolingtype == 0:
			pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
		else:
			pool1 = AveragePooling3D(pool_size=(2,2,2))(conv1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2)
		conv2 = Dropout(dropout)(conv2)	
		if poolingtype == 0:
			pool3 = MaxPooling3D(pool_size=(2,2,2))(conv2)
		else:
			pool3 = AveragePooling3D(pool_size=(2,2,2))(conv2)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3)
		conv3 = Dropout(dropout)(conv3)	
		if poolingtype == 0:
			pool4 = MaxPooling3D(pool_size=(2,2,2))(conv3)
		else:
			pool4 = AveragePooling3D(pool_size=(2,2,2))(conv3)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4)
		conv4 = Dropout(dropout)(conv4)	
		if poolingtype == 0:
			pool5 = MaxPooling3D(pool_size=(2,2,2))(conv4)
		else:
			pool5 = AveragePooling3D(pool_size=(2,2,2))(conv4)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5)
		conv5 = Dropout(dropout)(conv5)
		if poolingtype == 0:
			pool5a = MaxPooling3D(pool_size=(2,2,2))(conv5)
		else:
			pool5a = AveragePooling3D(pool_size=(2,2,2))(conv5)
		conv5a = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5a)
		conv5a = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5a)
		conv5a = Dropout(dropout)(conv5a)
	
		if poolingtype == 0:
			pool5ab = MaxPooling3D(pool_size=(2,2,2))(conv5a)
		else:
			pool5ab = AveragePooling3D(pool_size=(2,2,2))(conv5a)
	
		conv5ab = Conv3D(64*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5ab)
		conv5ab = Conv3D(64*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5ab)
		conv5ab = Dropout(dropout)(conv5ab)

		up6ab  = UpSampling3D(size=(2,2,2))(conv5ab)
		up6ab   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6ab)

		merge6ab = concatenate([conv5a,up6ab])
		conv6ab = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6ab)
		conv6ab = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6ab)
		
		up6a   = UpSampling3D(size=(2,2,2))(conv6ab)
		up6a   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6a)
	
			

		merge6a = concatenate([conv5,up6a])
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6a)
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6a)

		merge6a = concatenate([conv5,up6a])
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6a)
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6a)


		up6   = UpSampling3D(size=(2,2,2))(conv6a)
		merge6 = concatenate([conv4,up6])
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6)
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6)
		
		up7   = UpSampling3D(size=(2,2,2))(conv6)
		up7   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up7)
		merge7 = concatenate([conv3,up7]) 
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7)
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8)
		up8   = UpSampling3D(size=(2,2,2))(conv8)
		merge8 = concatenate([conv2,up8]) 
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8)
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9)
		
		up9   = UpSampling3D(size=(2,2,2))(conv9)
		up9   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="SYMMETRIC")(up9)
		merge9 = concatenate([conv1,up9]) 
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9)
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		#conv10 = Conv3D(size,2, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		conv10 = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10) 
		model = Model(inputs = myinput, outputs = conv10) 
	if modelId == 4: # 129-129-180 
		myinput = Input((phiSlice,rRow,zColumn,1)) 
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1)
		conv1 = Dropout(dropout)(conv1)	
		if poolingtype == 0:
			pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)
		else:
			pool1 = AveragePooling3D(pool_size=(2,2,2))(conv1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1)
		conv2 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2)
		conv2 = Dropout(dropout)(conv2)	
		if poolingtype == 0:
			pool3 = MaxPooling3D(pool_size=(2,2,2))(conv2)
		else:
			pool3 = AveragePooling3D(pool_size=(2,2,2))(conv2)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3)
		conv3 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3)
		conv3 = Dropout(dropout)(conv3)	
		if poolingtype == 0:
			pool4 = MaxPooling3D(pool_size=(2,2,2))(conv3)
		else:
			pool4 = AveragePooling3D(pool_size=(2,2,2))(conv3)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4)
		conv4 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4)
		conv4 = Dropout(dropout)(conv4)	
		if poolingtype == 0:
			pool5 = MaxPooling3D(pool_size=(2,2,2))(conv4)
		else:
			pool5 = AveragePooling3D(pool_size=(2,2,2))(conv4)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5)
		conv5 = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5)
		conv5 = Dropout(dropout)(conv5)
		if poolingtype == 0:
			pool5a = MaxPooling3D(pool_size=(2,2,2))(conv5)
		else:
			pool5a = AveragePooling3D(pool_size=(2,2,2))(conv5)
		conv5a = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5a)
		conv5a = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5a)
		conv5a = Dropout(dropout)(conv5a)
	
		if poolingtype == 0:
			pool5ab = MaxPooling3D(pool_size=(2,2,2))(conv5a)
		else:
			pool5ab = AveragePooling3D(pool_size=(2,2,2))(conv5a)
	
		conv5ab = Conv3D(64*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5ab)
		conv5ab = Conv3D(64*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5ab)
		conv5ab = Dropout(dropout)(conv5ab)

		if poolingtype == 0:
			pool5abc = MaxPooling3D(pool_size=(2,2,2))(conv5ab)
		else:
			pool5abc = AveragePooling3D(pool_size=(2,2,2))(conv5ab)

		conv5abc = Conv3D(128*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5abc)
		conv5abc = Conv3D(128*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5abc)
		conv5abc = Dropout(dropout)(conv5abc)
		up6abc  = UpSampling3D(size=(2,2,2))(conv5abc)
		#up6abc   = SymmetricPadding3D(padding=((1,0),(0,0),(0,1)),mode='SYMMETRIC')(up6abc)

		merge6abc= concatenate([conv5ab,up6abc])



		conv6abc = Conv3D(64*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6abc)
		conv6abc = Conv3D(64*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6abc)

		up6ab  = UpSampling3D(size=(2,2,2))(conv6abc)
		up6ab   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode='SYMMETRIC')(up6ab)

		merge6ab = concatenate([conv5a,up6ab])
		conv6ab = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6ab)
		conv6ab = Conv3D(32*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6ab)
		
		up6a   = UpSampling3D(size=(2,2,2))(conv6ab)
		up6a   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6a)
		

		merge6a = concatenate([conv5,up6a])
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6a)
		conv6a = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6a)


		up6   = UpSampling3D(size=(2,2,2))(conv6a)
		merge6 = concatenate([conv4,up6])
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6)
		conv6 = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6)
		up7   = UpSampling3D(size=(2,2,2))(conv6)
		up7   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up7)
		merge7 = concatenate([conv3,up7]) 
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7)
		conv8 = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8)
		up8   = UpSampling3D(size=(2,2,2))(conv8)
		merge8 = concatenate([conv2,up8]) 
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8)
		conv9 = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9)
		up9   = UpSampling3D(size=(2,2,2))(conv9)
		up9   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="SYMMETRIC")(up9)
		merge9 = concatenate([conv1,up9]) 
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9)
		conv10 = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10)
		conv10 = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10) 
		model = Model(inputs = myinput, outputs = conv10) 
	# 3 channels 90 - 17 -17
	if modelId == 5:
		# input charge
		myinput = Input((phiSlice,rRow,zColumn,1))
		conv1R = Conv3D(size,3, padding="same", kernel_initializer ="he_uniform")(myinput)
		if (batch_normalization):
			conv1R = BatchNormalization()(conv1R)
		conv1R = Activation('relu')(conv1R)
		conv1R = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="he_uniform")(conv1R)
		if (batch_normalization):
			conv1R = BatchNormalization()(conv1R)
		conv1R = Dropout(dropout)(conv1R)	
		if poolingtype == 0:
			pool1R = MaxPooling3D(pool_size=(2,2,2))(conv1R)
		else:
			pool1R = AveragePooling3D(pool_size=(2,2,2))(conv1R)
		conv2R = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1R)
		conv2R = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2R)
		conv2R = Dropout(dropout)(conv2R)	
		if poolingtype == 0:
			pool3R = MaxPooling3D(pool_size=(2,2,2))(conv2R)
		else:
			pool3R = AveragePooling3D(pool_size=(2,2,2))(conv2R)
		conv3R = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3R)
		conv3R = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3R)
		conv3R = Dropout(dropout)(conv3R)	
		if poolingtype == 0:
			pool4R = MaxPooling3D(pool_size=(2,2,2))(conv3R)
		else:
			pool4R = AveragePooling3D(pool_size=(2,2,2))(conv3R)
		conv4R = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4R)
		conv4R = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4R)
		conv4R = Dropout(dropout)(conv4R)	
		if poolingtype == 0:
			pool5R = MaxPooling3D(pool_size=(2,2,2))(conv4R)
		else:
			pool5R = AveragePooling3D(pool_size=(2,2,2))(conv4R)
		conv5R = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5R)
		conv5R = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5R)
		conv5R = Dropout(dropout)(conv5R)	
		up6R   = UpSampling3D(size=(2,2,2))(conv5R)
		up6R   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6R)
		merge6R = concatenate([conv4R,up6R]) 
		conv6R = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6R)
		conv6R = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6R)
		up7R   = UpSampling3D(size=(2,2,2))(conv6R)
		merge7R = concatenate([conv3R,up7R]) 
		conv8R = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7R)
		conv8R = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8R)
		up8R   = UpSampling3D(size=(2,2,2))(conv8R)
		up8R   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up8R)
		merge8R = concatenate([conv2R,up8R]) 
		conv9R = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8R)
		conv9R = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9R)
		up9R   = UpSampling3D(size=(2,2,2))(conv9R)
		up9R   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="CONSTANT")(up9R)
		merge9R = concatenate([conv1R,up9R]) 
		conv10R = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9R)
		conv10R = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10R)
		conv10R = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10R) 
		conv1RPhi = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1RPhi = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1RPhi)
		conv1RPhi = Dropout(dropout)(conv1RPhi)	
		if poolingtype == 0:
			pool1RPhi = MaxPooling3D(pool_size=(2,2,2))(conv1RPhi)
		else:
			pool1RPhi = AveragePooling3D(pool_size=(2,2,2))(conv1RPhi)
		conv2RPhi = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1RPhi)
		conv2RPhi = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2RPhi)
		conv2RPhi = Dropout(dropout)(conv2RPhi)	
		if poolingtype == 0:
			pool3RPhi = MaxPooling3D(pool_size=(2,2,2))(conv2RPhi)
		else:
			pool3RPhi = AveragePooling3D(pool_size=(2,2,2))(conv2RPhi)
		conv3RPhi = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3RPhi)
		conv3RPhi = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3RPhi)
		conv3RPhi = Dropout(dropout)(conv3RPhi)	
		if poolingtype == 0:
			pool4RPhi = MaxPooling3D(pool_size=(2,2,2))(conv3RPhi)
		else:
			pool4RPhi = AveragePooling3D(pool_size=(2,2,2))(conv3RPhi)
		conv4RPhi = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4RPhi)
		conv4RPhi = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4RPhi)
		conv4RPhi = Dropout(dropout)(conv4RPhi)	
		if poolingtype == 0:
			pool5RPhi = MaxPooling3D(pool_size=(2,2,2))(conv4RPhi)
		else:
			pool5RPhi = AveragePooling3D(pool_size=(2,2,2))(conv4RPhi)
		conv5RPhi = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5RPhi)
		conv5RPhi = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5RPhi)
		conv5RPhi = Dropout(dropout)(conv5RPhi)	
		up6RPhi   = UpSampling3D(size=(2,2,2))(conv5RPhi)
		up6RPhi   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6RPhi)
		merge6RPhi = concatenate([conv4RPhi,up6RPhi]) 
		conv6RPhi = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6RPhi)
		conv6RPhi = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6RPhi)
		up7RPhi   = UpSampling3D(size=(2,2,2))(conv6RPhi)
		merge7RPhi = concatenate([conv3RPhi,up7RPhi]) 
		conv8RPhi = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7RPhi)
		conv8RPhi = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8RPhi)
		up8RPhi   = UpSampling3D(size=(2,2,2))(conv8RPhi)
		up8RPhi   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up8RPhi)
		merge8RPhi = concatenate([conv2RPhi,up8RPhi]) 
		conv9RPhi = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8RPhi)
		conv9RPhi = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9RPhi)
		up9RPhi   = UpSampling3D(size=(2,2,2))(conv9RPhi)
		up9RPhi   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="CONSTANT")(up9RPhi)
		merge9RPhi = concatenate([conv1RPhi,up9RPhi]) 
		conv10RPhi = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9RPhi)
		conv10RPhi = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10RPhi)
		conv10RPhi = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10RPhi) 
		conv1Z = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(myinput)
		conv1Z = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv1Z)
		conv1Z = Dropout(dropout)(conv1Z)	
		if poolingtype == 0:
			pool1Z = MaxPooling3D(pool_size=(2,2,2))(conv1Z)
		else:
			pool1Z = AveragePooling3D(pool_size=(2,2,2))(conv1Z)
		conv2Z = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool1Z)
		conv2Z = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv2Z)
		conv2Z = Dropout(dropout)(conv2Z)	
		if poolingtype == 0:
			pool3Z = MaxPooling3D(pool_size=(2,2,2))(conv2Z)
		else:
			pool3Z = AveragePooling3D(pool_size=(2,2,2))(conv2Z)
		conv3Z = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool3Z)
		conv3Z = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv3Z)
		conv3Z = Dropout(dropout)(conv3Z)	
		if poolingtype == 0:
			pool4Z = MaxPooling3D(pool_size=(2,2,2))(conv3Z)
		else:
			pool4Z = AveragePooling3D(pool_size=(2,2,2))(conv3Z)
		conv4Z = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool4Z)
		conv4Z = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv4Z)
		conv4Z = Dropout(dropout)(conv4Z)	
		if poolingtype == 0:
			pool5Z = MaxPooling3D(pool_size=(2,2,2))(conv4Z)
		else:
			pool5Z = AveragePooling3D(pool_size=(2,2,2))(conv4Z)
		conv5Z = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(pool5Z)
		conv5Z = Conv3D(16*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv5Z)
		conv5Z = Dropout(dropout)(conv5Z)	
		up6Z   = UpSampling3D(size=(2,2,2))(conv5Z)
		up6Z   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up6Z)
		merge6Z = concatenate([conv4Z,up6Z]) 
		conv6Z = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge6Z)
		conv6Z = Conv3D(8*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv6Z)
		up7Z   = UpSampling3D(size=(2,2,2))(conv6Z)
		merge7Z = concatenate([conv3Z,up7Z]) 
		conv8Z = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge7Z)
		conv8Z = Conv3D(4*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv8Z)
		up8Z   = UpSampling3D(size=(2,2,2))(conv8Z)
		up8Z   = SymmetricPadding3D(padding=((1,0),(0,0),(0,0)),mode="SYMMETRIC")(up8Z)
		merge8Z = concatenate([conv2Z,up8Z]) 
		conv9Z = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge8Z)
		conv9Z = Conv3D(2*size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv9Z)
		up9Z   = UpSampling3D(size=(2,2,2))(conv9Z)
		up9Z   = SymmetricPadding3D(padding=((0,0),(1,0),(1,0)),mode="CONSTANT")(up9Z)
		merge9Z = concatenate([conv1Z,up9Z]) 
		conv10Z = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(merge9Z)
		conv10Z = Conv3D(size,3, activation="relu", padding="same", kernel_initializer ="normal")(conv10Z)
		conv10Z = Conv3D(1,1, activation="linear", kernel_initializer="normal")(conv10Z)
		output = concatenate([conv10R,conv10RPhi,conv10Z]) 
		model = Model(inputs = myinput, outputs = output) 
	return model	


# read TChain or TFile save each in separated dataframe
def GetLocalCurrentNumpyArrayToNpy(fnames,phi_slice=180,r_row=65,z_col=65):
	vecFluctuationSC = []
	vecFluctuationDistR = []
	step = 0
	fluctuationDir = os.environ['FLUCTUATIONDIR']
	baseDataDir = fluctuationDir + "data/"
	dataDir = baseDataDir + str(phi_slice) + '-' + str(r_row) + '-' + str(z_col) + '/'
	os.mkdir(dataDir)
	ID = 0
	for fname in fnames:
		if isfile(fname + '.h5'):
			print("found h5 file")
			hf = h5py.File(fname + '.h5', 'r')
			dataSet = np.array(hf["meanCurrent"][:])
			print(dataSet[0][:])
			print("len : " + str(len(dataSet[0][:])))
			vecFluctuationSC = dataSet[0][:]
			vecFluctuationDistR = dataSet[1][:]
			for vecIn,vecOut in zip(vecFluctuationSC,vecFluctuationDistR):
				np.save(dataDir  + str(ID) + '-scFluctuation.npy',vecIn)
				np.save(dataDir  + str(ID) + '-distRFluctuation.npy',vecOut)
				ID = ID + 1
				print(str(ID))	
		
	return ID		


def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv3D(dim, 3, activation=acti, padding='same', kernel_initializer="normal")(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv3D(dim, 3, activation=acti, padding='same', kernel_initializer="normal")(n)
	n = BatchNormalization()(n) if bn else n
	return concatenate([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, pool_type, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		if (pool_type == 0):
			m = MaxPooling3D(pool_size=(2,2,2))(n) 
		elif (pool_type == 1):
			m = AveragePooling3D(pool_size=(2,2,2))(n) 
		else:
			Conv3D(dim, 3, strides=2, padding='same')(n)

		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, pool_type, up, res)

		if up:
			m = UpSampling3D(size=(2,2,2))(m)
			diff_phi = n.shape[1] - m.shape[1]
			diff_r = n.shape[2] - m.shape[2]
			diff_z = n.shape[3] - m.shape[3]
			if (diff_phi != 0):
				m = SymmetricPadding3D(padding=((int(diff_phi),0),(int(diff_r),0),(int(diff_z),0)),mode="SYMMETRIC")(m)
			elif ((diff_r !=0) or (diff_z != 0)):
				m = SymmetricPadding3D(padding=((int(diff_phi),0),(int(diff_r),0),(int(diff_z),0)),mode="CONSTANT")(m)
			
		#	m = Conv3D(dim, 3, activation=acti, padding='same',kernel_initializer="normal")(m)
		else:
            		m = Conv3DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = concatenate([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

# Get U-net model:
def UNet(input_shape,start_ch=4,depth=4,inc_rate=2.0,activation="relu",dropout=0.2,bathnorm=False,pool_type=0,upconv=True,residual=False):
	i = Input(shape=input_shape)
	output_r 	= level_block(i,start_ch,depth,inc_rate,activation,dropout,bathnorm,pool_type,upconv,residual)
	output_r 	= Conv3D(1,1, activation="linear",padding="same",kernel_initializer="normal")(output_r)

	output_rphi 	= level_block(i,start_ch,depth,inc_rate,activation,dropout,bathnorm,pool_type,upconv,residual)
	output_rphi 	= Conv3D(1,1, activation="linear", padding="same", kernel_initializer="normal")(output_rphi)


	output_z 	= level_block(i,start_ch,depth,inc_rate,activation,dropout,bathnorm,pool_type,upconv,residual)
	output_z	= Conv3D(1,1, activation="linear",padding="same", kernel_initializer="normal")(output_z)
	o = concatenate([output_r,output_rphi,output_z])
	return Model(inputs=i,outputs=output_r)


def GetFluctuation(phiSlice,rRow,zColumn,id,side=0):
	"""
	Get fluctuation id
	"""
	fluctuationDir = os.environ['FLUCTUATIONDIR']
	dataDir = fluctuationDir + 'data/' + str(phiSlice) + '-' + str(rRow) + '-' + str(zColumn) + '/'
	vecZPosFile  = dataDir + str(0) + '-vecZPos.npy'
	scMeanFile = dataDir + str(id) + '-vecMeanSC.npy'
	scRandomFile = dataDir + str(id) + '-vecRandomSC.npy'
	distRMeanFile = dataDir + str(id) + '-vecMeanDistR.npy'
	distRRandomFile = dataDir + str(id) + '-vecRandomDistR.npy'
	distRPhiMeanFile = dataDir + str(id) + '-vecMeanDistRPhi.npy'
	distRPhiRandomFile = dataDir + str(id) + '-vecRandomDistRPhi.npy'
	distZMeanFile = dataDir + str(id) + '-vecMeanDistZ.npy'
	distZRandomFile = dataDir + str(id) + '-vecRandomDistZ.npy'
	vecZPos = np.load(vecZPosFile)
	vecMeanSC = np.load(scMeanFile)
	vecRandomSC = np.load(scRandomFile)
	vecMeanDistR = np.load(distRMeanFile)
	vecRandomDistR = np.load(distRRandomFile)
	vecMeanDistRPhi = np.load(distRPhiMeanFile)
	vecRandomDistRPhi = np.load(distRPhiRandomFile)
	vecMeanDistZ = np.load(distZMeanFile)
	vecRandomDistZ = np.load(distZRandomFile)
	vecFluctuationSC = vecMeanSC[vecZPos >= 0] - vecRandomSC[vecZPos >= 0]
	vecFluctuationDistR = vecMeanDistR[vecZPos >= 0] - vecRandomDistR[vecZPos >= 0]
	vecFluctuationDistRPhi = vecMeanDistRPhi[vecZPos >= 0] - vecRandomDistRPhi[vecZPos >= 0]
	vecFluctuationDistZ= vecMeanDistZ[vecZPos >= 0] - vecRandomDistZ[vecZPos >= 0]
	return [vecFluctuationSC,vecFluctuationDistR,vecFluctuationDistRPhi,vecFluctuationDistZ]


import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.externals import joblib


def GetScaler(phiSlice, rRow, zColumn,numberOfSample=1, side = 0):
	"""
	Get Scaler for preprocessing
	"""
	
	sc = np.array([])
	distR = np.array([])
	distRPhi = np.array([])
	distZ = np.array([])
	for i in range(0,numberOfSample):
	#	id = random.randint(0,1000)
	#	print(str(id))
		print(str(i))
		[vecFluctuationSC,vecFluctuationDistR,vecFluctuationDistRPhi,vecFluctuationDistZ] = GetFluctuation(phiSlice,rRow,zColumn,i)
		sc = np.concatenate([sc,vecFluctuationSC])
		distR = np.concatenate([distR,vecFluctuationDistR])
		distRPhi = np.concatenate([distRPhi,vecFluctuationDistRPhi])
		distZ = np.concatenate([distZ,vecFluctuationDistZ])
	
	scalerSC = StandardScaler()
	scalerDistR = StandardScaler()
	scalerDistRPhi = StandardScaler()
	scalerDistZ = StandardScaler()

	scalerSC.fit(sc.reshape(-1,1))
	scalerDistR.fit(distR.reshape(-1,1))
	scalerDistRPhi.fit(distRPhi.reshape(-1,1))
	scalerDistZ.fit(distZ.reshape(-1,1))
	fluctuationDir = os.environ['FLUCTUATIONDIR']
	dataDir = fluctuationDir + 'data/' + str(phiSlice) + '-' + str(rRow) + '-' + str(zColumn) + '/'

	joblib.dump(scalerSC, dataDir + "scalerSC.save")	
	joblib.dump(scalerDistR, dataDir + "scalerDistR.save")	
	joblib.dump(scalerDistRPhi, dataDir + "scalerDistRPhi.save")	
	joblib.dump(scalerDistZ, dataDir + "scalerDistZ.save")	
	return [scalerSC,scalerDistR,scalerDistRPhi,scalerDistZ]


#model = UNet((90,17,17,1),bathnorm=True,pool_type=1)
#model.summary()	
#GetScaler(90,17,17,numberOfSample=1000)


