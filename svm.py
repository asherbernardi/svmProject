import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix
from sklearn.metrics.pairwise import rbf_kernel as sklrbf

def linear(x1, x2) :
	return 1 + np.dot(x1, x2)

def make_poly_kernel(s) :
	return lambda x1, x2 : (1 + np.dot(x1, x2))**s

def rbf(x1, x2) :
	return sklrbf([x1], [x2])[0][0]

class SVM:
	def __init__(self, data, targets, k, threshold, C=None):
		assert(len(data) == len(targets))
		self.N = len(data)
		self.data = np.array(data)
		self.targets = np.array(targets)
		self.kernel = k
		self.C = C
		self.trained = False
		self.threshold = threshold
	def train(self):
		# reasign for convenience
		N = self.N
		data = self.data
		targets = self.targets
		C = self.C
		# begin algorithm
		K = np.array([[self.kernel(x_i, x_j) for x_i in data] for x_j in data])
		print("K is:\n" + str(K))
		P = targets * targets.T * K
		print("P is:\n" + str(P))
		q = np.full((N,1), -1)
		print("q is (matrix):\n" + str(matrix(q)))
		print("q.shape" + str(q.shape))
		A = targets.reshape(1,N)
		print("A is:\n" + str(A))
		G = -np.identity(N)
		print("G is:\n" + str(G))
		h = np.zeros((N,1))
		print("h is:\n" + str(h))
		# for soft margin classification
		if C != None:
			G = np.concatenate(np.identity(N), G)
			h = np.concatenate(np.full((N,1), C), h)
		solved = qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'), matrix(A, (1,N), tc='d'), matrix(0, tc='d'))
		a = np.array(solved['x'])
		print("a computed is:\n" + str(a))
		a_adj = [ai for ai in a if ai != 0];
		self.b = np.reciprocal(len(a_adj)) * np.sum([targets[j] - np.sum([a[i] * targets[i] * self.kernel(data[i], data[j]) for i in range(N) if a[i] != 0]) for j in range(N) if a[j] != 0])
		self.a = a
		self.trained = True

	def classify(self, inputs):
		assert(self.trained)
		results = [0]*len(inputs)
		print("results: " + str(results))
		for r,x in enumerate(inputs):
			for i in range(self.N):
				if self.a[i] != 0:
					print("data[i]:" + str(self.data[i]))
					print("x:" + str(x))
					print(self.a[i][0])
					print(self.targets[i])
					print(self.kernel(self.data[i], x))
					results[r] += self.a[i][0] * self.targets[i] * self.kernel(self.data[i], x)
			results[r] += self.b
			print("results[r]: " + str(results[r]))
		return results

def train(data, targets, k, threshold=1e-5, C=None):
	clsfyr = SVM(data, targets, k, C)
	clsfyr.train()
	return clsfyr

def classify(svm, inputs):
	return svm.classify(inputs)
