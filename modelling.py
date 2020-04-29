import os
import sys


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from keras.callbacks.callbacks import ModelCheckpoint
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Sequential,Model
from keras.layers import Input,Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score
import pickle


class Modelling:

	def __init__(self):

		pass

	def load_data(self,df):

		self.ret_arr = []
		for i in df['Names']:
			self.img = load_img(i[1:],target_size=(128,128))
			self.img = img_to_array(self.img)/255.
			self.ret_arr.append(self.img)

		return np.array(self.ret_arr)

	def show_img(self,path):
		self.img = load_img(path[1:],target_size=(128,128))
		self.img = img_to_array(self.img)
		cv2.imshow(self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		return


	def sample_model(self):

		self.model = Sequential()
		self.model.add(Conv2D(128,(3,3),input_shape=(128,128,3),activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(2,2))
		self.model.add(Dropout(0.25))
		self.model.add(Conv2D(64,(3,3),activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(64,(3,3),activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(2,2))
		self.model.add(Dropout(0.25))
		self.model.add(Conv2D(128,(3,3),activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(2,2))
		self.model.add(Conv2D(128,(3,3),activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.5))
		self.model.add(Flatten())
		self.model.add(Dense(1024,activation='relu'))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(128,activation='relu'))
		self.model.add(Dense(8,activation='softmax'))

		return self.model






	pass


def main():
	np.random.seed(1998)
	util = Modelling()
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	valid = pd.read_csv('valid.csv')

	train = train.sample(frac=1)
	test = test.sample(frac=1)
	valid = valid.sample(frac=1)

	train = train[:1300]
	valid = valid[:400]
	test = test[:400]

	y_train = train['Class']
	y_test = test['Class']
	y_valid = valid['Class']
	y_train_cat = to_categorical(y_train)[:,1:]
	y_test_cat = to_categorical(y_test)[:,1:]
	y_valid_cat = to_categorical(y_valid)[:,1:]


	train_data = util.load_data(train)
	test_data = util.load_data(test)
	valid_data = util.load_data(valid)


	print(train_data.shape)
	print(test_data.shape)
	print(valid_data.shape)


	sample_model = util.sample_model()
	print(sample_model.summary())

	checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', verbose=0, 
		save_best_only=True, save_weights_only=False, mode='auto', period=1)


	x_train,x_test,y_train,y_test = train_test_split(train_data,y_train_cat,test_size=0.3)

	sample_model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
	history = sample_model.fit(train_data,y_train_cat,epochs=50,
		validation_data=(valid_data,y_valid_cat),verbose=1,
		callbacks=[checkpoint])


	print(sample_model.evaluate(test_data,y_test_cat))

	accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1,len(accuracy)+1)

	fig = plt.figure(figsize=(20,10))
	plt.subplot(2,2,1)
	plt.plot(epochs,accuracy,'b',label='Training_Accuracy')
	plt.plot(epochs,val_accuracy,'r',label='Validation_Accuracy')
	plt.title('Training and Validation Accuracy')
	plt.legend()

	plt.subplot(2,2,2)
	plt.plot(epochs,loss,'b',label='Training_Loss')
	plt.plot(epochs,val_loss,'r',label='Validation_Loss')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.show()
	pickle.dump(sample_model,open('cnn_model.pkl', 'wb'))
	pass


if __name__ == "__main__":
	main()