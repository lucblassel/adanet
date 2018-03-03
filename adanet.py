# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2018-02-28 15:38:45
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-02 14:54:09
import numpy as np
from keras.layers import Input, Dense, concatenate
from keras.models import Model, load_model
from keras.utils import plot_model
from keras import backend as k
from keras import optimizers
from keras.datasets import cifar10
import dataProcessing as dp
import copy as cp
from itertools import chain
import dill
from pprint import pprint
import inspect

def runthrough(T,depth,layerDic):
	print(100*"*")
	for rep in range(depth):
		for t in range(T):
			concatName = 'c'+str(rep)+'.'+str(t)
			layerName = str(rep)+'.'+str(t)
			print(concatName,layerName)
			try:
				print(layerDic[concatName])
			except:
				pass
			try:
				print(layerDic[layerName])
			except:
				pass

def toSymbolicDict(T,depth,layerDic):
	"""
	luc.blassel@agroparistech.fr
	"""
	tensorDic = {}
	print('layerDic:')
	pprint(layerDic)
	print()
	key = "feeding.Layer"
	params = layerDic[key]
	tensorDic[key] = Input(shape=params[1]['shape'],name=key)

	for rep in range(depth):
		for t in range(T):
			for prefix in ("c",""):
				key = prefix+str(rep)+'.'+str(t)
				try:
					params = layerDic[key]
					if key[0] == 'c': #concatenating layer
						print("bulding concat layer")
						print(params[1])
						candidateLayers = layerCall(tensorDic,params[1])
						tensorDic[key] = params[0](candidateLayers)
					elif key != 'output.Layer':
						print('tensorDic')
						pprint(tensorDic)
						print(key,params[2])
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
	tensorDic[key] = params[0](params[1]['units'],activation=params[1]['activation'],name=key)(tensorDic[params[2]])

	return tensorDic


def builderNew(B,T,flattenDimIm,lr,reps,x_train,y_train,x_test,y_test,epochs,batch_size,NrandomModels,epsilon,pathToSaveModel,proba_threshold):
	"""
	luc.blassel@agroparistech.fr
	"""
	layerDic = {}
	layersNamesToOutput = []
	concatOutName = 'c.out'

	layerDic['feeding.Layer'] = (Input,{'shape':(flattenDimIm,),'name':'feeding.Layer'})

	for t in range(T):
		print('\n\n'+100*"="+'\niteration n.'+str(t)+'\n'+100*"=")
		if t==0:
			layerName = '0.0'
			layerDic[layerName] = (Dense,{'units':B,'activation':'relu','name':layerName},'feeding.Layer')
			layerDic['output.Layer'] = (Dense,{'units':1,'activation':'sigmoid','name':'output.Layer'},layerName)
			layersNamesToOutput.append(layerName)
			previousScore = 10000


			symbolicTensorsDict = toSymbolicDict(1,1,layerDic)
			model = Model(inputs=symbolicTensorsDict['feeding.Layer'],outputs=symbolicTensorsDict['output.Layer'])
			model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
			model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,verbose=1)
			model.save_weights('w_'+pathToSaveModel)
			model.save(pathToSaveModel)

			plot_model(model,to_file='modelIt1.png',show_shapes=True)



			with open('layerDic.pkl','wb') as dicFile:
				dill.dump(layerDic,dicFile)
			with open('layersNamesToOutput.pkl','wb') as outFile:
				dill.dump(layersNamesToOutput,outFile)
			k.clear_session()
		else:
			for rep in range(reps):
				print('\n rep '+str(rep))

				modelTest = load_model(pathToSaveModel)
				previousDepth = getPreviousDepth(layerDic,t)
				previousPredictions = classPrediction(modelTest,x_test,proba_threshold)

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
					if depth == 0 :
						layerDic[layerName] = (Dense,{'units':B,'activation':'relu','name':layerName},'feeding.Layer')
					else:
						candidateNameList = selectCandidateLayers(layerDic,t,depth)
						candidateNameList = drawing(candidateNameList)
						layerBelowName = str(depth-1)+'.'+str(t)
						candidateNameList.append(layerBelowName)
						candidateNameList = list(set(candidateNameList))
						if len(candidateNameList)>1:
							layerDic[concatLayerName] = (concatenate,candidateNameList)
							layerDic[layerName] = (Dense,{'units':B,'activation':'relu','name':layerName},concatLayerName)
						else :
							layerDic[layerName] = (Dense,{'units':B,'activation':'relu','name':layerName},candidateNameList[0])
					if depth == currentDepth-1:
						layersNamesToOutput.append(layerName)
				# layersToOutput = layerCall(layerDic,layersNamesToOutput)
				if len(layersNamesToOutput)>1 :
					layerDic[concatOutName] = (concatenate,list(set(layersNamesToOutput)))
					layerDic['output.Layer'] = (Dense,{'units':1,'activation':'sigmoid','name':'output.Layer'},concatOutName)
				else:
					layerDic['output.Layer'] = (Dense,{'units':1,'activation':'sigmoid','name':'output.Layer'},layersNamesToOutput[0])

				runthrough(t+1,currentDepth+1,layerDic)
				symbolicTensorsDict = toSymbolicDict(t+1,currentDepth,layerDic)

				model = Model(inputs=symbolicTensorsDict['feeding.Layer'], outputs=symbolicTensorsDict['output.Layer'])
				model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

				print()
				pprint([layer.name for layer in model.layers])

				plot_model(model,to_file="modelIt"+str(t)+"Rep"+str(rep)+'.png',show_shapes=True)

				model.layers[-1].name += 'temp'
				model.load_weights('w_'+pathToSaveModel,by_name=True)
				model.layers[-1].name = model.layers[-1].name[:-4]
				model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,verbose=1)

				currentPredictions = classPrediction(model,x_test,proba_threshold)
				currentScore = objectiveFunction(y_test,previousPredictions,currentPredictions)
				if previousScore - currentScore > epsilon:
					print("saving better model")
					previousScore = currentScore
					model.save('best'+pathToSaveModel)
					model.save_weights('best_w_'+pathToSaveModel)
					with open('layersNamesToOutput.pkl', 'wb') as f:
						dill.dump(layersNamesToOutput, f)
					with open('layerDic.pkl', 'wb') as f:
						dill.dump(layerDic, f)
				k.clear_session()


# def builder(B,T,flattenDimIm,lr,reps,x_train,y_train,x_test,y_test,epochs,batch_size,NrandomModels,epsilon,pathToSaveModel,proba_threshold):
# 	"""
# 	romain.gautron@agroparistech.fr
# 	"""
# 	layerDic = {}
# 	layersNamesToOutput = []
# 	concatOutName= 'c.output'
# 	layerDic['feeding.Layer'] = Input(shape=(flattenDimIm,),name='feeding.Layer')
# 	for t in range(T):
# 		if t == 0:
# 			layerName = '0.0'
# 			layerDic[layerName] = Dense(B, activation='relu',name=layerName)(layerDic['feeding.Layer'])
# 			layerDic['output.Layer'] = Dense(1, activation='sigmoid',name='output.Layer')(layerDic[layerName])
# 			layersNamesToOutput.append(layerName)
# 			previousScore = 10000
# 			model = Model(inputs=layerDic['feeding.Layer'], outputs=layerDic['output.Layer'])
# 			model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
# 			model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,verbose=1)
# 			model.save(pathToSaveModel)
# 			with open('layersNamesToOutput.pkl', 'wb') as f:
# 				dill.dump(layersNamesToOutput, f)
# 			with open('layerDic.pkl', 'wb') as f:
# 				dill.dump(layerDic, f)
# 			k.clear_session()
# 		else:
# 			model = load_model(pathToSaveModel)
# 			previousDepth = getPreviousDepth(layerDic,t)
# 			previousPredictions = classPrediction(model,x_test,proba_threshold)
# 			for rep in range(reps):
# 				with open('layerDic.pkl', 'rb') as f:
# 					layerDic = dill.load(f)
# 				with open('layersNamesToOutput.pkl', 'rb') as f:
# 					layersNamesToOutput = dill.load(f)
# 				if rep > reps//2 :
# 					currentDepth = previousDepth
# 					print('same layer')
# 				else :
# 					currentDepth = previousDepth + 1
# 					print('going deeper')
# 				for depth in range (currentDepth):
# 					layerName = str(depth)+'.'+str(t)
# 					concatLayerName = 'c' + layerName
# 					if depth == 0 :
# 						layerDic[layerName]=Dense(B, activation='relu',name=layerName)(layerDic['feeding.Layer'])
# 					else:
# 						candidateNameList = selectCandidateLayers(layerDic,t,depth)
# 						candidateNameList = drawing(candidateNameList)
# 						layerBelowName = str(depth-1)+'.'+str(t)
# 						candidateNameList.append(layerBelowName)
# 						candidateNameList = list(set(candidateNameList))
# 						candidateLayers = layerCall(layerDic,candidateNameList)
# 						if len(candidateLayers)>1:
# 							layerDic[concatLayerName] = concatenate(candidateLayers)
# 							layerDic[layerName]=Dense(B, activation='relu',name=layerName)(layerDic[concatLayerName])
# 						else :
# 							layerDic[layerName]=Dense(B, activation='relu',name=layerName)(candidateLayers[0])
# 					if depth == currentDepth-1:
# 						layersNamesToOutput.append(layerName)
# 				layersToOutput = layerCall(layerDic,layersNamesToOutput)
# 				if len(layersToOutput)>1 :
# 					layerDic[concatOutName] = concatenate(layersToOutput)
# 					layerDic['output.Layer'] = Dense(1, activation='sigmoid',name='output.Layer')(layerDic[concatOutName])
# 				else:
# 					layerDic['output.Layer'] = Dense(1, activation='sigmoid',name='output')(layersToOutput[0])
# 				model = Model(inputs=layerDic['feeding.Layer'], outputs=layerDic['output.Layer'])
# 				model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
# 				model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,verbose=1)
#
# 				plot_model(model,to_file="modelFin.png",show_shapes=True)
# 				currentPredictions = classPrediction(model,x_test,proba_threshold)
# 				currentScore = objectiveFunction(y_test,previousPredictions,currentPredictions)
# 				if previousScore - currentScore > epsilon:
# 					print("better model found")
# 					previousScore = currentScore
# 					model.save(pathToSaveModel)
# 					# model.save_weights('w_'+pathToSaveModel)
# 					with open('layersNamesToOutput.pkl', 'wb') as f:
# 						dill.dump(layersNamesToOutput, f)
# 					with open('layerDic.pkl', 'wb') as f:
# 						dill.dump(layerDic, f)
# 				k.clear_session()

# saving layerNamesToOut
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


def classPrediction(model,x,proba_threshold):
	probas = np.array(model.predict(x))
	booleans = probas >= proba_threshold
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

def main():
	pathToSaveModel = 'bestModel.h5'
	imsize =  32
	flattenDimIm = imsize*imsize*3
	B = 10
	T = 10
	lr = .0001
	reps = 5
	trainNum = 5000
	testNum = 10
	epochs = 10
	batch_size = 100
	NrandomModels  = 10
	epsilon = .001
	labels = [0,1]
	proba_threshold = .5

	train, test = dp.loadRawData()
	x_train, y_train = dp.loadTrainingData(train,labels,trainNum)
	x_test, y_test = dp.loadTestingData(test,labels,testNum)

	x_train_reshaped = x_train.flatten().reshape(trainNum,flattenDimIm)/255
	x_test_reshaped = x_test.flatten().reshape(testNum,flattenDimIm)/255

	builderNew(B,T,flattenDimIm,lr,reps,x_train_reshaped,y_train[:trainNum],x_test_reshaped,y_test,epochs,batch_size,NrandomModels,epsilon,pathToSaveModel,proba_threshold)

	# layer = Input(shape=(flattenDimIm,),name="input.layer")
	# layer2 = Dense(B,activation = 'relu',name='0.0')(layer)
    #
	# pprint(vars(layer))
	# pprint(vars(layer2))
    #
	# pprint(inspect.getmembers(layer))
	# builder(B,T,flattenDimIm,lr,1,x_train_reshaped,y_train[:trainNum],x_test_reshaped,y_test,epochs,batch_size,NrandomModels,epsilon,pathToSaveModel,proba_threshold)
	# plot_model(model,to_file='model.png',show_shapes=True)


	# preds = model.predict(x_test_reshaped)
    #
	# for i in range(len(preds)):
	# 	print(preds[i],y_test[i])

if __name__ == '__main__':
	main()
