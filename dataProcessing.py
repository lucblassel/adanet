"""
2018-01-14T21:49:15.920Z
-----------------------------------------------
BLASSEL Luc
Data processing
-----------------------------------------------
"""
import keras
from keras.datasets import cifar10
import numpy as np
from scipy.misc import imresize
from scipy.misc import imshow
import pickle
import os
from sklearn.model_selection import train_test_split

# resizeFactor = 1

def load_class_names():
	"""
	Load the names for the classes in the CIFAR-10 data-set.

	Returns a list with the names. Example: names[3] is the name
	associated with class-number 3.
	(taken from cifar10.py, modified by luc blassel)
	"""

	print("Loading data: " + file_path)

	with open(file_path, mode='rb') as file:
		# In Python 3.X it is important to set the encoding,
		# otherwise an exception is raised here.
		meta = pickle.load(file, encoding='bytes')

	# Load the class-names from the pickled file.
	raw = meta[b'label_names']

	# Convert from binary strings.
	names = [x.decode('utf-8') for x in raw]

	return names

def getLabelIndexes(labels):
	"""
	gets label int from string
	"""
	if type(labels[0]) == str:
		ind = np.zeros((len(labels)),dtype=np.int)
		names = load_class_names()
		c = 0

		for i in range(len(names)):
			if names[i] in labels:
				ind[c] = i
				c += 1
		return ind
	elif type(labels[0]) == int:
		return labels
	else:
		raise ValueError("labels must be either string or index numbers")

def loadRawData():
	"""
	loads default keras CIFAR-10 dataset
	"""
	train,test = cifar10.load_data()
	return train,test

def loadTrainingData(train,labels,trainCases,**kwargs):
	"""
	loads a training set of size trainCases, and of the classes from the labels array
	"""

	if 'resizeFactor' in kwargs:
		resizeFactor = kwargs['resizeFactor']
	else:
		resizeFactor = 1

	#initialising arrays for better performance
	sub_x_train = np.zeros((trainCases,resizeFactor*32,resizeFactor*32,3),dtype=np.int)
	sub_y_train = np.zeros((trainCases),dtype=np.int)

	x_train = train[0]
	y_train = train[1]

	ind = getLabelIndexes(labels)

	c = 0
	for i in range(len(y_train)):
		if y_train[i][0] in ind:
			if resizeFactor > 1:
				sub_x_train[c] = imresize(x_train[i],resizeFactor*100,'nearest')
			else:
				sub_x_train[c] = x_train[i]
			sub_y_train[c] = y_train[i]
			c += 1
			if c >= trainCases:
				break
	return sub_x_train,sub_y_train

def loadTestingData(test,labels,testCases,**kwargs):
	"""
	loads a testing set of size testCases, and of classes from the labels arrray
	"""

	if 'resizeFactor' in kwargs:
		resizeFactor = kwargs['resizeFactor']
	else:
		resizeFactor = 1


	#initialising arrays for better performance
	sub_x_test = np.zeros((testCases,resizeFactor*32,resizeFactor*32,3),dtype=np.int)
	sub_y_test = np.zeros((testCases),dtype=np.int)

	x_test = test[0]
	y_test = test[1]

	ind = getLabelIndexes(labels)

	c = 0
	for i in range(len(y_test)):
		if y_test[i][0] in ind:
			if resizeFactor > 1:
				sub_x_test[c] = imresize(x_test[i],resizeFactor*100,'nearest')
			else:
				sub_x_test[c] = x_test[i]
			sub_y_test[c] = y_test[i]
			c += 1
			if c >= testCases:
				break

	return sub_x_test,sub_y_test

def validationSplitter(x,y,proportion,shuffle):
	"""
	returns x_train,x_val,y_train,y_val
	if shuffle is true training set is shuffled before splitting
	"""
	return train_test_split(x,y,proportion,shuffle=shuffle)

def main():
	trainLabels = ['dog','truck']
	testLabels = ['dog','truck']
	trainnum = 1000
	testnum = 1000

	train, test = loadRawData()

	x_train,y_train = loadTrainingData(train,[0,1],trainnum)
	x_test,y_test = loadTestingData(test,[0,1],testnum)

	print(y_train,y_test)


if __name__ == "__main__":
	main()
