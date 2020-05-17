from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Autoencoders():

	def __init__(self):

		pass

	def modelDef(self):
		self.input_img = Input(shape=(128, 128, 3)) 


		self.x = Conv2D(16, (3, 3), activation='relu', padding='same')(self.input_img)
		self.x = MaxPooling2D((2, 2), padding='same')(x)
		self.x = Conv2D(8, (3, 3), activation='relu', padding='same')(self.x)
		self.x = MaxPooling2D((2, 2), padding='same')(x)
		self.x = Conv2D(8, (3, 3), activation='relu', padding='same')(self.x)
		self.encoded = MaxPooling2D((2, 2), padding='same')(self.x)

		

		self.x = Conv2D(8, (3, 3), activation='relu', padding='same')(self.encoded)
		self.x = UpSampling2D((2, 2))(self.x)
		self.x = Conv2D(8, (3, 3), activation='relu', padding='same')(self.x)
		self.x = UpSampling2D((2, 2))(self.x)
		self.x = Conv2D(16, (3, 3), activation='relu')(self.x)
		self.x = UpSampling2D((2, 2))(self.x)
		self.decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(self.x)

		self.autoencoder = Model(self.input_img, self.decoded)

		return self.autoencoder

	def trainModel(self,model,x_train):
		model.compile(optimizer='adadelta', loss='binary_crossentropy')
		model.fit(x_train, x_train,
	                epochs=50,
	                batch_size=128,
	                shuffle=True,
	                validation_data=(x_test, x_test),
	                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

		return model.predict(x_train)