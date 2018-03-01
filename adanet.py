# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2018-02-28 15:38:45
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-03-01 17:05:47
import numpy as np
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.utils import plot_model
from keras import backend as k
from keras import optimizers
from keras.datasets import cifar10
import dataProcessing as dp

def builder(B,T,flattenDimIm,lr,option,reps,x_train,y_train,epochs,batch_size):
	"""
	romain.gautron@agroparistech.fr
	"""
	layerDic = {}
	layersNamesToOutput = []
	concatOutName= 'c.output'
	layerDic['feeding.Layer'] = Input(shape=(flattenDimIm,),name='feeding.Layer')
	for t in range(T):
		if t == 0:
			layerName = '0.0'
			layerDic[layerName] = Dense(B, activation='relu',name=layerName)(layerDic['feeding.Layer'])
			layerDic['output.Layer'] = Dense(1, activation='sigmoid',name='output.Layer')(layerDic[layerName])
			layersNamesToOutput.append(layerName)
			model = Model(inputs=layerDic['feeding.Layer'], outputs=layerDic['output.Layer'])
			model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
		else:
			PreviousDepth = getPreviousDepth(layerDic,t)
			if option == "A":
				currentDepth = PreviousDepth
			else :
				currentDepth = PreviousDepth + 1
			for depth in range (currentDepth):
				layerName = str(depth)+'.'+str(t)
				concatLayerName = 'c' + layerName
				if depth == 0 :
					layerDic[layerName]=Dense(B, activation='relu',name=layerName)(layerDic['feeding.Layer'])
				else:
					candidateNameList = selectCandidateLayers(layerDic,t,depth)
					candidateNameList = drawing(candidateNameList)
					layerBelowName = str(depth-1)+'.'+str(t)
					candidateNameList.append(layerBelowName)
					candidateNameList = list(set(candidateNameList))
					candidateLayers = layerCall(layerDic,candidateNameList)
					if len(candidateLayers)>1:
						layerDic[concatLayerName] = concatenate(candidateLayers)
						layerDic[layerName]=Dense(B, activation='relu',name=layerName)(layerDic[concatLayerName])
					else :
						layerDic[layerName]=Dense(B, activation='relu',name=layerName)(candidateLayers[0])
				if depth == currentDepth-1:
					layersNamesToOutput.append(layerName)
			layersToOutput = layerCall(layerDic,layersNamesToOutput)
			if len(layersToOutput)>1 :
				layerDic[concatOutName] = concatenate(layersToOutput)
				layerDic['output.Layer'] = Dense(1, activation='sigmoid',name='output.Layer')(layerDic[concatOutName])
			else:
				layerDic['output.Layer'] = Dense(1, activation='sigmoid',name='output')(layersToOutput[0])
			model = Model(inputs=layerDic['feeding.Layer'], outputs=layerDic['output.Layer'])
			model.compile(optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

		model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=batch_size,verbose=1)
	return model, layerDic

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

def main():

	imsize =  32
	flattenDimIm = imsize*imsize*3
	B = 10
	T = 10
	lr = .0001
	trainNum = 5000
	testNum = 10
	epochs = 100
	batch_size = 100

	labels = [0,2]

	train, test = dp.loadRawData()
	x_train, y_train = dp.loadTrainingData(train,labels,trainNum)
	x_test, y_test = dp.loadTestingData(test,labels,testNum)

	x_train_reshaped = x_train.flatten().reshape(trainNum,flattenDimIm)/255
	x_test_reshaped = x_test.flatten().reshape(testNum,flattenDimIm)/255

	print(x_train_reshaped.shape)

	model,layerDic = builder(B,T,flattenDimIm,lr,"B",1,x_train_reshaped,y_train[:trainNum],epochs,batch_size)
	plot_model(model,to_file='model.png',show_shapes=True)


	preds = model.predict(x_test_reshaped)

	for i in range(len(preds)):
		print(preds[i],y_test[i])

if __name__ == '__main__':
	main()
