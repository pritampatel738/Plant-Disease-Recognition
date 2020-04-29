import os
import math
import sys
import pandas as pd
import cv2

class renameTool:


	def __init__(self):

		pass

	def generateCsv(self):
		"""

			To generate a csv file in order to load images to training and testing......

		"""
		print("=========================================================")
		self.dirs = os.getcwd()
		self.className = 0
		self.target = []
		self.file = []

		for (self.dname,self.dirs,self.files) in os.walk("."):

			if self.dname == "__init.py__":
				continue
			
			for filename in self.files:
				if filename.endswith(".JPG"):
					self.file.append(self.dname[1:]+"/"+str(filename))
					self.target.append(self.className)
			self.className += 1

			pass
		print("The total number of targets are : ",len(self.target))
		print("The total number of files are : ",len(self.file))

		self.df = pd.DataFrame({"Names":self.file,"Class":self.target})
		self.df.to_csv("data.csv",index=False)

		self.df = self.df.sample(frac=1)
		print(self.df.head())
		print(self.df['Class'].value_counts())
		print("=============================================================")

	pass

	pass



	def createTrainTestValidation(self):
		"""
		
		To divide the dataset into train validation and test set....

		"""
		print("=========================================================")
		df = pd.read_csv('data.csv')
		print("Data is read successfully.")
		
		"""
		path = df['Names'].iloc[0]
		print("The path is {}.".format(path))
		img = cv2.imread(path[1:],1)
		print("Img read successfully.")
		print(type(img))
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		"""

		class1 = df[df['Class']==1]
		class2 = df[df['Class']==2]
		class3 = df[df['Class']==3]
		class4 = df[df['Class']==4]
		class5 = df[df['Class']==5]
		class6 = df[df['Class']==6]
		class7 = df[df['Class']==7]
		class8 = df[df['Class']==8]

		traindf = pd.DataFrame()
		validdf = pd.DataFrame()
		testdf = pd.DataFrame()

		traindf = class1[:220]
		validdf = class1[220:300]
		testdf = class1[300:]

		traindf = traindf.append(class2[:220])
		validdf = validdf.append(class2[220:300])
		testdf = testdf.append(class2[300:])

		traindf = traindf.append(class3[:220])
		validdf = validdf.append(class3[220:300])
		testdf = testdf.append(class3[300:])

		traindf = traindf.append(class4[:220])
		validdf = validdf.append(class4[220:300])
		testdf = testdf.append(class4[300:])

		traindf = traindf.append(class5[:220])
		validdf = validdf.append(class5[220:300])
		testdf = testdf.append(class5[300:])

		traindf = traindf.append(class6[:220])
		validdf = validdf.append(class6[220:300])
		testdf = testdf.append(class6[300:])

		traindf = traindf.append(class7[:220])
		validdf = validdf.append(class7[220:300])
		testdf = testdf.append(class7[300:])

		traindf = traindf.append(class8[:220])
		validdf = validdf.append(class8[220:300])
		testdf = testdf.append(class8[300:])

		print("The shape of train is : ",traindf.shape)
		print("The shape of valid is : ",validdf.shape)
		print("The shape of test is : ",testdf.shape)

		traindf.to_csv('train.csv',index=False)
		validdf.to_csv('valid.csv',index=False)
		testdf.to_csv('test.csv',index=False)

		print("=========================================================")

		pass

	def rename(self):

		"""

		to rename the file according to our convenience....
		"""

		self.sysPath = os.getcwd()
		self.totalCount = 0
		for (self.dname,self.dirs,self.files) in os.walk("."):
			if self.dname == "__init__.py":
				continue
			self.count = 0
			print(self.dname)
			print(self.dirs)
			print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			self.arrtbr = []
			for filename in self.files:
				if filename.endswith(".JPG"):
					self.arrtbr.append(int(filename[:-4]))
			self.arrtbr = sorted(self.arrtbr)
			#print(arrtbr)
			#break	
			for filename in self.arrtbr:
				#print(self.sysPath+"/"+self.dname[2:]+"/"+str(filename)+".JPG"+"  "+self.sysPath+"/"+self.dname[2:]+"/"+str(self.count)+".JPG")
				os.rename(self.sysPath+"/"+self.dname[2:]+"/"+str(filename)+".JPG",self.sysPath+"/"+self.dname[2:]+"/"+str(self.count)+".JPG")
				self.count += 1
				self.totalCount += 1
			print("{} contains {} images.".format(self.dname,self.count))


		print("The total number of images are : ",self.totalCount)

def main():
	
	rename = renameTool()
	rename.rename()
	rename.generateCsv()
	rename.createTrainTestValidation()




if __name__ == "__main__":
	main()















































































































































