# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2018-02-28 15:38:45
# @Last Modified by:   Romain
# @Last Modified time: 2018-02-28 20:13:59
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras import backend as k

def builder(B,T,flattenDimIm,option,reps):
	layersNamesToOutput = []
	feedingLayer = Input(shape=(flattenDimIm,),name='feedingLayer')
	for t in range(T):
		if t == 0:
			layerName = '0.0'
			globals()[layerName]=Dense(B, activation='relu',name='layerName')(feedingLayer)
			layersNamesToOutput.append(layerName)
			layersToOutput = list(map(layerCall,layersNamesToOutput))
			outputLayer = Dense(1, activation='sigmoid',name='output')(layersToOutput)
			model = Model(inputs=feedingLayer, outputs=outputLayer)
			model.compile(optimizer = optimizers.Adam(lr=lr_target,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
		else:
			PreviousDepth = getPreviousDepth(model,t)
			if option == "A":
				currentDepth = getPreviousDepth
			else :
				currentDepth = getPreviousDepth + 1
				for depth in range (currentDepth):
					layerName = str(c)+'.'+str(t)
					if depth == 0 : 
						globals()[layerName]=Dense(B, activation='relu',name='layerName')(feedingLayer)
					else:
						#for rep in range(reps):
						candidateList = selectCandidateLayers(model,t,c)
						layerBelowName = str(c-1)+'.'+str(t)
						candidateList.append(layerBelowName)
						globals()[layerName]=Dense(B, activation='relu',name='layerName')(candidateList)
						if depth == currentDepth-1:
							layersNamesToOutput.append(layerName)
			layersToOutput = list(map(layerCall,layersNamesToOutput))
			outputLayer = Dense(1, activation='sigmoid',name='output')(layersToOutput)
			model = Model(inputs=feedingLayer, outputs=outputLayer)
			model.compile(optimizer = optimizers.Adam(lr=lr_target,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
	return model


def drawing(candidatNames):
	numberToDraw = random.randint(0, len(candidatNames))
	result = np.random.choice(candidatNames, size=numberToDraw, replace=False)
	return result

def getPreviousDepth(model,t):
	previousDepth = 0
	for layer in model.layers:
		depth,iteration = layer.name.split('.')
		if iteration == t-1:
			if depth > previousDepth:
				previousDepth = depth
	return previousDepth + 1

def selectCandidateLayers(model,t,c):
	candidateList = []
	for layer in model.layers:
		depth,iteration = layer.name.split('.')
		if iteration < t and depth == c-1 :
			candidateList.append(layer.name)
	return candidateList

def layerCall(name):
	return globals()[name]

def main():
	imsize =  32
	flattenDimIm = imsize*imsize*3
	B = 10
	T = 5
	model = builder(B,T,flattenDimIm,"A",1)
	plot_model(model,to_file='model.png',show_shapes=True)

if __name__ == '__main__':
	main()