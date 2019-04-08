import numpy as np
from cvxopt.solvers import qp
from sklearn.metrics.pairwise import rbf_kernel as sklrbf

def linear(x1, x2) :
    return 1 + np.dot(x1, x2)

def make_poly_kernel(s) :
    return lambda x1, x2 : (1 + np.dot(x1, x2))**s

def rbf(x1, x2) :
    return sklrbf([x1], [x2])[0][0]

class svm:
	def __init__(self, data, targets, k, C=None):
		assert(len(data) = len(targets))
		self.N = len(data)
		self.data = data
		self.targets = targets
		self.kernel = k
		self.C = C
		self.trained = False
	def train(self):
		K = np.array([[self.kernel(data[i], data[j]) for i in range(N)] for j in range(N)])
		P = np.multiply(np.dot(self.targets, self.targets), K)
		q = np.full([N,1],-1)
		A = np.multiply(np.transpose(targets), np.identity(N))
	def classify(self, x):
		assert(self.trained)
		result = 0
		for i in range(self.N):
			if self.a[i] != 0:
				result += a[i] * targets[i] * self.kernel(data[i], x)
		result += b
		return result

def train(data, targets, k, C=None):


def classify(svm, inputs):