import numpy as np, mpmath as mp
from sklearn.metrics import accuracy_score

class AdalineGD(object):
	def __init__(self, eta=0.01, n_iter=1000):
		self.eta = eta
		self.n_iter = n_iter

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		return self.net_input(X)

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1])
		self.errors_ = []
		self.cost_ = []

		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			for index in range(errors.size):
				errors[index] = mp.power(errors[index], 2)
			cost = errors.sum() / 2.0
			self.cost_.append(cost)
		return self

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1)

	def score(self, X, Y, sample_weight=None):
		return accuracy_score(Y, self.predict(X), sample_weight=sample_weight)