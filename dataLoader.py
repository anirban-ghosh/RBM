import numpy as np

class dataLoader(object):
	"""docstring for dataLoader"""
	def __init__(self, trainFile, valFile):
		super(dataLoader, self).__init__()
		self.trainFile = trainFile
		self.valFile = valFile
		with open(self.trainFile, 'r') as f:
			rawDataSet = np.genfromtxt(f, dtype='float32', delimiter=',')
		self.numTrain = rawDataSet.shape[0]
		vecDim = rawDataSet.shape[1]
		xDim = 28
		yDim = 28
		self.imageVectorSize = xDim * yDim
		self.labelsTrain = rawDataSet[:,vecDim - 1].astype(np.uint8)
		self.dataTrain = rawDataSet[:,0:vecDim - 1]

		with open(self.valFile, 'r') as f:
		   rawDataSetVal = np.genfromtxt(f, dtype='float32', delimiter=',')

		self.numValid = rawDataSetVal.shape[0]
		vecDimVal = rawDataSetVal.shape[1]

		self.labelsVal = rawDataSetVal[:,vecDimVal - 1].astype(np.uint8)
		self.dataVal = rawDataSetVal[:,0:vecDimVal - 1]

	def shuffleTrain(self):
		shuffle = np.random.permutation(self.dataTrain.shape[0])
		self.dataTrain = self.dataTrain[shuffle]
		self.labelsTrain = self.labelsTrain[shuffle]

	def getNumTrain(self):
		return self.numTrain
	def getNumVal(self):
		return self.numValid
	def getTrainSample(self, i):
		return self.dataTrain[i].reshape((1, self.imageVectorSize)), self.labelsTrain[i]
	def getValSample(self, i):
		return self.dataVal[i].reshape((1, self.imageVectorSize)), self.labelsVal[i]
	def getVectorDim(self):
		return self.imageVectorSize