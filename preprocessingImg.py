import os
import pandas as pd
import numpy as np
import cv2
from keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt









def main():


	df = pd.read_csv('data.csv')
	print(df.head())

	img = load_img(df['Names'].iloc[15][1:],target_size=(100,100))
	img = img_to_array(img)/255.
	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	pass






if __name__ == "__main__":
	main()
