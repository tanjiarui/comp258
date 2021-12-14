import numpy as np, json
from sklearn import preprocessing
from sklearn.metrics import classification_report

sigmoid = lambda x: 1 / (1 + np.exp(-x))
def backprop(W1, W2, X, Y, alpha=.2):
	for b in range(X.shape[0]):
		x = X[b, :].T  # inputs from training data
		y = Y[b]  # correct output from training data
		##########################
		# forward propagation step
		##########################
		# calculate the weighted sum of hidden node
		v1 = np.dot(W1, x)
		# pass the weighted sum to the activation function, this gives the outputs from hidden layer
		y1 = sigmoid(v1)
		# calculate the weighted sum of the output layer
		v = np.dot(W2, y1)
		# pass it to the activation function, this returns the output of the third layer
		y_hat = sigmoid(v)
		# calculate the error, difference between correct output and computed output
		error = y - y_hat
		# calculate delta, derivative of the activation function times the error
		# note that 𝜎′(𝑥)=𝜎(𝑥)∙(1− 𝜎(𝑥)) = y * (1-y)
		delta = y_hat * (1 - y_hat) * error  # element wise multiplication
		###########################
		# Backward propagation step
		# Stochastic Gradient Descent
		###########################
		# propagate the output node delta, δ, backward, and calculate the deltas of the hidden layer.
		e1 = np.dot(W2.T, delta)
		delta1 = y1 * (1 - y1) * e1  # element wise multiplication

		# Adjust the weights according to the learning rule
		delta1 = delta1.reshape(delta1.size, 1)
		x = x.reshape(1, x.size)
		dW1 = alpha * np.dot(delta1, x)
		W1 = W1 + dW1
		delta = delta.reshape(delta.size, 1)
		y1 = y1.reshape(1, y1.size)
		dW2 = alpha * np.dot(delta, y1)
		W2 = W2 + dW2

	return W1, W2

def train(X, y, node, lr=.2, epoch=10000):
	input_size, output_size = X.shape[1], np.unique(y).size
	# initialize the weights between input layer and hidden layer
	W1 = 2 * np.random.rand(node, input_size) - 1
	# initialize the weights between hidden layer and output layer
	W2 = 2 * np.random.rand(output_size, node) - 1
	for e in range(epoch): # train
		W1, W2 = backprop(W1, W2, X, y, lr)
	return W1, W2

def predict(X, W1, W2):
	y = []
	for b in range(X.shape[0]):
		x = X[b, :].T
		# calculate the weighted sum of hidden node
		v1 = np.dot(W1, x)
		# pass the weighted sum to the activation function, this gives the outputs from hidden layer
		y1 = sigmoid(v1)
		# calculate the weighted sum of the output layer
		v = np.dot(W2, y1)
		# pass it to the activation function, this returns the output of the third layer
		y.append(np.argmax(sigmoid(v)))
	return y

with open('iris.json', 'r', encoding='utf-8-sig') as file:
	iris = json.load(file)
for index in range(len(iris)):
	iris[index] = list(iris[index].values())
iris = np.array(iris)
X, y = iris[:, :4].astype('float'), iris[:, 4]
y = preprocessing.LabelEncoder().fit_transform(y)
for alpha in np.arange(0, .5, .02):
	w1, w2 = train(X, y, 5, alpha)
	prediction = predict(X, w1, w2)
	print('learning rate: ' + str(alpha))
	print(classification_report(y, prediction))
