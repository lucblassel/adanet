# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2018-02-28 15:38:45
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-09 19:36:53
import numpy as np
import keras
from keras.layers import Input, Dense, concatenate, add
from keras.models import Model, load_model
from keras.utils import plot_model
from keras import backend as k
from keras import optimizers
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, Callback
from keras.regularizers import l1
import dataProcessing as dp
import copy as cp
from itertools import chain
import dill
from pprint import pprint
import inspect
import os
from shutil import copyfile
import time

class StopEarly(Callback):
	def __init__(self,threshold,metric="val_acc",verbose = True):
		super(StopEarly,self).__init__()
		self.threshold = threshold
		self.metric = metric
		self.last_value = 0
		self.stopped_epoch = 0
		self.verbose = verbose

	def on_epoch_end(self, epoch, logs={}):
		current = logs.get(self.metric)
		if logs.get(self.metric) - self.last_value < self.threshold:
			self.model.stop_training = True
			self.stopped_epoch = epoch
		self.last_value = current

	def on_train_end(self, log={}):
		if self.stopped_epoch > 0 and self.verbose:
			print("model stopped training on epoch",self.stopped_epoch,"with val_acc =",self.last_value)


def runthrough(T,depth,layerDic):
	print('\n\nrunthrough'+100*"*")
	for rep in range(depth):
		for t in range(T):
			for prefix in ("c",""):
				name = prefix+str(rep)+"."+str(t)
				try:
					print(name,layerDic[name])
				except:
					pass
	for name in ("c.out",'output.Layer'):
		try:
			print(name,layerDic[name])
		except:
			pass
	print('\n\n')

def toSymbolicDict(T,depth,layerDic):
	"""
	luc.blassel@agroparistech.fr
	"""
	tensorDic = {}
	# print('layerDic:')
	# pprint(layerDic)
	# print()
	key = "feeding.Layer"
	params = layerDic[key]
	tensorDic[key] = Input(shape=params[1]['shape'],name=key)

	for rep in range(depth):
		for t in range(T):
			for prefix in ("c",""):
				key = prefix+str(rep)+'.'+str(t)
				try:
					params = layerDic[key]
					# print("\n\n\nconsidered layer:",key)
					if key[0] == 'c': #concatenating layer
						# print("bulding concat layer")
						# print(params[1])
						candidateLayers = layerCall(tensorDic,params[1])
						tensorDic[key] = params[0](candidateLayers)
					elif key != 'output.Layer':
						# print()
						# pprint(layerDic)
						# print('tensorDic')
						# pprint(tensorDic)
						# print(params)
						tensorDic[key] = params[0](params[1]['units'],activation=params[1]['activation'],name=key)(tensorDic[params[2]])
				except:
					pass


	key = 'c.out'
	try :
		params = layerDic[key]
		candidateLayers = layerCall(tensorDic,params[1])
		tensorDic[key] = params[0](candidateLayers)
	except:
		pass


	key = 'output.Layer'
	params = layerDic[key]
	# pprint(layerDic)
	# print('tensorDic')
	# pprint(tensorDic)
	tensorDic[key] = params[0](params[1]['units'],activation=params[1]['activation'],name=key)(tensorDic[params[2]])

	return tensorDic


def builderNew(B,T,flattenDimIm,lr,reps,xTrain,yTrain,xTest,yTest,epochs,batchSize,NrandomModels,epsilon,pathToSaveModel,probaThreshold,handleMultipleInput,lambda1):
	"""
	luc.blassel@agroparistech.fr
	"""
	count = 1
	layerDic = {}
	layersNamesToOutput = []
	concatOutName = 'c.out'

	earlyStopping = StopEarly(0.001,"val_acc",True)

	layerDic['feeding.Layer'] = (Input,{'shape':(flattenDimIm,),'name':'feeding.Layer'})

	for t in range(T):
		changed = False #boolean to track if the base model is changed (improved)
		print('\n\n'+100*"="+'\niteration n.'+str(t)+'\n'+100*"=")
		if t==0:
			layerName = '0.0'
			layerDic[layerName] = (Dense,{'units':B,'activation':'relu','kernel_regularizer':l1(lambda1),'name':layerName},'feeding.Layer')
			layerDic['output.Layer'] = (Dense,{'units':1,'activation':'sigmoid','kernel_regularizer':l1(lambda1),'name':'output.Layer'},layerName)
			layersNamesToOutput.append(layerName)
			previousScore = float('Inf')

			symbolicTensorsDict = toSymbolicDict(1,1,layerDic)
			model = Model(inputs=symbolicTensorsDict['feeding.Layer'],outputs=symbolicTensorsDict['output.Layer'])
			model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
			model.fit(x=xTrain,y=yTrain,validation_split=0.1,callbacks=[earlyStopping],epochs=epochs,batch_size=batchSize,verbose=1)
			model.save_weights('w_'+pathToSaveModel)
			model.save(pathToSaveModel)

			plot_model(model,to_file='modelIt1.png',show_shapes=True)

			with open('layerDic.pkl','wb') as dicFile:
				dill.dump(layerDic,dicFile)
			with open('layersNamesToOutput.pkl','wb') as outFile:
				dill.dump(layersNamesToOutput,outFile)
			k.clear_session()
		else:

			if t>1:
				copyfile(pathToSaveModel,str(t-1)+pathToSaveModel)
				copyfile('w_'+pathToSaveModel,'w_'+str(t-1)+pathToSaveModel)
				try:
					os.rename('best_'+pathToSaveModel,pathToSaveModel)
					os.rename('best_w_'+pathToSaveModel,'w_'+pathToSaveModel)
				except:
					pass

			for rep in range(reps):
				print('\n rep '+str(rep))

				modelTest = load_model(pathToSaveModel)
				plot_model(modelTest,to_file='modeltest'+str(rep)+'.png',show_shapes=True)
				previousDepth = getPreviousDepth(layerDic,t)
				previousPredictions = classPrediction(modelTest,xTest,probaThreshold)

				with open('layerDic.pkl', 'rb') as f:
					layerDic = dill.load(f)
				with open('layersNamesToOutput.pkl', 'rb') as f:
					layersNamesToOutput = dill.load(f)
				if rep > reps//2 :
					currentDepth = previousDepth
					print("staying at current level")
				else :
					currentDepth = previousDepth + 1
					print("going deeper")

				for depth in range (currentDepth):
					layerName = str(depth)+'.'+str(t)

					concatLayerName = 'c' + layerName
					if handleMultipleInput == 'concatenate':
						functionChoice = concatenate
					elif handleMultipleInput == 'add':
						functionChoice = add
					else:
						raise ValueError("handleMultipleInput must have a value in ('concatenate','add')")

					if depth == 0 :
						layerDic[layerName] = (Dense,{'units':B,'activation':'relu','name':layerName},'feeding.Layer')
					else:
						candidateNameList = selectCandidateLayers(layerDic,t,depth)
						candidateNameList = drawing(candidateNameList)
						layerBelowName = str(depth-1)+'.'+str(t)
						candidateNameList.append(layerBelowName)
						candidateNameList = list(set(candidateNameList))
						if len(candidateNameList)>1:
							layerDic[concatLayerName] = (functionChoice,candidateNameList)
							layerDic[layerName] = (Dense,{'units':B,'activation':'relu','kernel_regularizer':l1(lambda1),'name':layerName},concatLayerName)
						else :
							layerDic[layerName] = (Dense,{'units':B,'activation':'relu','kernel_regularizer':l1(lambda1),'name':layerName},candidateNameList[0])
					if depth == currentDepth-1:
						layersNamesToOutput.append(layerName)

				if len(layersNamesToOutput)>1 :
					layerDic[concatOutName] = (functionChoice,list(set(layersNamesToOutput)))
					layerDic['output.Layer'] = (Dense,{'units':1,'activation':'sigmoid','kernel_regularizer':l1(lambda1),'name':'output.Layer'},concatOutName)
				else:
					layerDic['output.Layer'] = (Dense,{'units':1,'activation':'sigmoid','kernel_regularizer':l1(lambda1),'name':'output.Layer'},layersNamesToOutput[0])

				# runthrough(t+1,currentDepth+1,layerDic)
				symbolicTensorsDict = toSymbolicDict(t+1,currentDepth+1,layerDic)

				model = Model(inputs=symbolicTensorsDict['feeding.Layer'], outputs=symbolicTensorsDict['output.Layer'])
				model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

				#input size of output layer changes so loading pre-existing weights is not possible
				if handleMultipleInput == 'concatenate':
					model.layers[-1].name += 'temp'
					model.load_weights('w_'+pathToSaveModel,by_name=True)
					model.layers[-1].name = model.layers[-1].name[:-4]
				else:
					model.load_weights('w_'+pathToSaveModel,by_name=True)

				model.fit(x=xTrain,y=yTrain,validation_split=0.1,callbacks=[earlyStopping],epochs=epochs,batch_size=batchSize,verbose=0)
				print("fitted model number ",count)
				count += 1
				currentPredictions = classPrediction(model,xTest,probaThreshold)
				currentScore = objectiveFunction(yTest,previousPredictions,currentPredictions)

				if previousScore - currentScore > epsilon:
					print("saving better model")
					changed = True
					plot_model(model,to_file="modelIt"+str(t)+"Rep"+str(rep)+'.png',show_shapes=True)
					previousScore = currentScore
					model.save('best_'+pathToSaveModel)
					model.save_weights('best_w_'+pathToSaveModel)
					with open('layersNamesToOutput.pkl', 'wb') as f:
						dill.dump(layersNamesToOutput, f)
					with open('layerDic.pkl', 'wb') as f:
						dill.dump(layerDic, f)
				k.clear_session()
				print("\n\n currentScore: ",currentScore,'\n\n')
			if not changed:
				print("model not improved at iteration",t,"stopping early")
				return
	bestModel = load_model('best_'+pathToSaveModel)
	print(bestModel.metric_names)
	print("Test metrics : ",bestModel.evaluate(xTest,yTest,verbose=0))
	k.clear_session()

def drawing(candidatNames):
	numberToDraw = np.random.randint(0, len(candidatNames))
	result = np.random.choice(candidatNames, size=numberToDraw, replace=False)
	return result.tolist()

def getPreviousDepth(layerDic,t):
	previousDepth = 0
	for layerName in layerDic.keys():
		depth,iteration = layerName.split('.')
		try :
			depth_int,iteration_int = int(depth),int(iteration)
			if iteration_int == t-1 and depth_int > previousDepth:
				previousDepth = depth_int
		except:
			pass
	return previousDepth + 1

def selectCandidateLayers(layerDic,t,c):
	candidateList = []
	for layerName in layerDic.keys():
		depth,iteration = layerName.split('.')
		try :
			depth_int,iteration_int = int(depth),int(iteration)
			if depth_int == c-1 :
				candidateList.append(layerName)
		except :
			pass
	return candidateList

def layerCall(dic,keys):
	return [dic[key] for key in keys]


def classPrediction(model,x,probaThreshold):
	probas = np.array(model.predict(x))
	booleans = probas >= probaThreshold
	booleans = list(chain(*booleans))
	classes = []
	for boolean in booleans:
		if boolean:
			classes.append(1)
		else:
			classes.append(-1)
	return classes

def objectiveFunction(trueLabels,previousPredictions,currentPredictions):

	m = len(trueLabels)
	result = 0
	for i in range(m):
		result += np.exp(1 - trueLabels[i]*previousPredictions[i]-trueLabels[i]*currentPredictions[i])
	result = result/m
	return result

def binaryEncoding(y_vect):
	"""
	encodes two classes as labels 0 and 1
	"""
	return np.array([0 if i==y_vect[0] else 1 for i in y_vect])

def oppositeEncoding(y_vect):
	"""
	encodes 0 and 1 as -1 and 1 for real values and predictions
	"""
	return np.array([-1 if int(round(i))==0 else 1 for i in y_vect])

def main():
	pathToSaveModel = 'bestModel.h5'
	imsize =  32
	flattenDimIm = imsize*imsize*3
	B = 150
	T = 10
	lr = .001
	reps = 5
	trainNum = 5000
	testNum = 1000
	epochs = 10000
	batchSize = 32
	NrandomModels  = 10
	epsilon = .0001
	labels = [1,2]
	probaThreshold = .5
	handleMultipleInput = "add"
	lambda1 = 0.000001

	if len(labels)>2 or labels[0]==labels[1]:
		raise ValueError('labels must be array of 2 distinct values')
	for i in range(2):
		if labels[i] <0 or labels[i]>9:
			raise ValueError('label value must be between 0 and 9 included')

	train, test = dp.loadRawData()
	xTrain, yTrain = dp.loadTrainingData(train,labels,trainNum)
	xTest, yTest = dp.loadTestingData(test,labels,testNum)

	xTrainReshaped = xTrain.flatten().reshape(trainNum,flattenDimIm)/255
	xTestReshaped = xTest.flatten().reshape(testNum,flattenDimIm)/255


	yTrain = binaryEncoding(yTrain)
	yTest = binaryEncoding(yTest)

	builderNew(B,T,flattenDimIm,lr,reps,xTrainReshaped,yTrain[:trainNum],xTestReshaped,yTest,epochs,batchSize,NrandomModels,epsilon,pathToSaveModel,probaThreshold,handleMultipleInput,lambda1)

	model = load_model(pathToSaveModel)

	plot_model(model,to_file="finalModel.png",show_shapes=True)

	preds = model.predict(xTestReshaped)
	error = 0
	for i in range(len(preds)):
		# print(int(np.round(preds[i])),yTest[i])
		if int(np.round(preds[i])) != yTest[i]:
			error +=1
	print("error:",error/testNum)


if __name__ == '__main__':
	main()
