import numpy as np

def sigmoid(x):
	# Vectorized numerically stable sigmoid calculator
	retVal = np.zeros(x.shape)
	z = np.zeros(x.shape)
	condition = x > 0
	z[np.where(condition)] = np.exp(-x[np.where(condition)])
	retVal[np.where(condition)] = 1/(1+z[np.where(condition)])
	z[np.where(~condition)] = np.exp(x[np.where(~condition)])
	retVal[np.where(~condition)] = z[np.where(~condition)]/(1+z[np.where(~condition)])
	return retVal

class RBM(object):
	"""docstring for RBM"""
	def __init__(self, 
		Input = None,
		visibleUnits = 784,
		hiddenUnits = 100,
		W = None, h_Bias = None, v_Bias = None):
		
		self.visibleUnits = visibleUnits
		self.hiddenUnits = hiddenUnits

		# Init weights and biases
		W = np.random.normal(loc = 0.0, scale = 0.1, size = (self.visibleUnits, self.hiddenUnits))
		h_Bias = np.zeros((self.hiddenUnits, 1))
		v_Bias = np.zeros((self.visibleUnits, 1))

		self.W = W
		self.h_Bias = h_Bias
		self.v_Bias = v_Bias
		self.Input = Input
		self.params = [self.W, self.h_Bias, self.v_Bias]

	def loadData(self, Input):
		self.Input = Input
	def forwardPass(self, visibleLayer):
		pre_activation = np.dot(visibleLayer, self.W) + self.h_Bias.T
		activation = sigmoid(pre_activation)
		return activation
	def sampleForward(self, visible_t0):
		activations = self.forwardPass(visible_t0)					# Find activations given visible at t=0
		hidden_t1 = np.random.binomial(n=1, p = activations, size=activations.shape)	# Sample hidden at t=1 based on activations
		return hidden_t1, activations				# Activations needed for xEntropy

	def reversePass(self, hiddenLayer):
		pre_activation = np.dot(hiddenLayer, self.W.T) + self.v_Bias.T
		activation = sigmoid(pre_activation)
		return activation
	def sampleReverse(self, hidden_t0):
		activations = self.reversePass(hidden_t0)
		visible_t1 = np.random.binomial(n=1, p = activations, size=activations.shape)
		return visible_t1, activations 				# Activations needed for xEntropy

	def getParams(self):
		return self.W, self.h_Bias, self.v_Bias

	def CD_K(self, K = 1, lr = 0.1, train=True):
		visible = self.Input
		hidden, h_Activations = self.sampleForward(visible)
		# Save initial hidden and visible states
		visible_t0 = visible 	
		hidden_t0 = hidden

		# Perfor K CD steps
		for i in range(K):						# For K CD steps, do
			visible, v_Activations = self.sampleReverse(hidden)		# Sample Visible layer given Hidden
			hidden, h_Activations = self.sampleForward(visible)		# Sample Hidden state given Visible 

		if(train == True):
			self.W = self.W + (lr * ((np.dot(hidden_t0.T, visible_t0)) - (np.dot(hidden.T, visible)))).T
			self.h_Bias = self.h_Bias + (lr * (hidden_t0 - hidden)).T
			self.v_Bias = self.v_Bias + (lr * (visible_t0 - visible)).T
		
		self.xEntropyLoss = -(np.mean(  
								np.multiply(visible_t0, np.log(v_Activations)) + 
								np.multiply((1 - visible_t0), np.log(1 - v_Activations)) 
								))
		return self.xEntropyLoss