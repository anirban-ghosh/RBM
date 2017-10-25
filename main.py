import numpy as np
import rbm
import dataLoader as dl
import matplotlib.pyplot as plt
import time


def visualize(layer, bias, numFilters, xDim, yDim):
	fig = plt.figure()
	for i in range(numFilters):
		featureMap = (layer.T[i] + bias.T[i]).reshape(xDim, yDim)
		ax = fig.add_subplot(10, 10, i+1)
		plt.imshow(featureMap, cmap = 'gray')
		ax.set_aspect('equal')
	plt.show()

def plotErrorGraph(trainPlot, valPlot):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Cross Entropy Loss v/s Training Epoch')
	plt.plot(np.arange(1, trainPlot.shape[0]+1), trainPlot, label="Training Loss")
	plt.plot(np.arange(1, trainPlot.shape[0]+1), valPlot, label="Validation Loss")
	axes = plt.gca()
	axes.set_xlim([0, trainPlot.shape[0]+1])
	plt.xlabel('Training Epoch')
	plt.ylabel('Cross Entropy Loss')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	loader = dl.dataLoader('digitstrain.txt', 'digitsvalid.txt')
	print "Data loaded"
	loader.shuffleTrain()
	network = rbm.RBM(visibleUnits = loader.getVectorDim(), hiddenUnits=100)
	print "RBM Initialized"
	numTrain = loader.getNumTrain()
	numValid = loader.getNumVal()
	numEpochs = 150
	K = 5
	lr = 0.1
	meanTrainLoss = np.zeros(numEpochs)
	meanValLoss = np.zeros(numEpochs)
	for epoch in range(numEpochs):
		if epoch == numEpochs/2:
			lr /= 10
		start = time.time()
		loader.shuffleTrain()
		trainLoss = np.zeros(numTrain)
		for i in range(numTrain):
			img, label = loader.getTrainSample(i)
			network.loadData(img)
			trainLoss[i] = network.CD_K(K=K, lr=lr, train=True)
		meanTrainLoss[epoch] = np.mean(trainLoss)
		valLoss = np.zeros(numValid)
		for i in range(numValid):
			img, label = loader.getValSample(i)
			network.loadData(img)
			valLoss[i] = network.CD_K(K=K, train = False)
		meanValLoss[epoch] = np.mean(valLoss)
		end = time.time()
		print 'Epoch: ', epoch, 'Train Loss:', ("%.4f" % meanTrainLoss[epoch]), 'Validation Loss: ',
		print ("%.4f" % meanValLoss[epoch]), 'Time: ', ("%.2f" % (end-start)) 
	
	plt.ion()
	plotErrorGraph(meanTrainLoss, meanValLoss)
	plt.pause(0.001)
	networkW, h_Bias, v_Bias = network.getParams()
	visualize(networkW, np.zeros(100), numFilters = 100, xDim = 28, yDim = 28)
	plt.pause(0.001)
	plt.show(block=True)

