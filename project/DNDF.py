import numpy as np, tensorflow as tf
from tensorflow.keras import Model

class NeuralDecisionTree(Model):
	def __init__(self, depth, num_features, used_features_rate, num_classes):
		super(NeuralDecisionTree, self).__init__()
		self.depth = depth
		self.num_leaves = 2 ** depth
		self.num_classes = num_classes
		# Create a mask for the randomly selected features.
		num_used_features = int(num_features * used_features_rate)
		one_hot = np.eye(num_features)
		sampled_feature_indicies = np.random.choice(np.arange(num_features), num_used_features, replace=False)
		self.used_features_mask = one_hot[sampled_feature_indicies]
		# Initialize the weights of the classes in leaves.
		self.pi = tf.Variable(initial_value=tf.random_normal_initializer()(shape=[self.num_leaves, self.num_classes]), dtype="float32", trainable=True)
		# Initialize the stochastic routing layer.
		self.decision_fn = tf.keras.layers.Dense(units=self.num_leaves, activation="sigmoid", name="decision")

	def call(self, features, training=None, mask=None):
		batch_size = tf.shape(features)[0]

		# Apply the feature mask to the input features.
		features = tf.matmul(features, self.used_features_mask, transpose_b=True)  # [batch_size, num_used_features]
		# Compute the routing probabilities.
		decisions = tf.expand_dims(self.decision_fn(features), axis=2)  # [batch_size, num_leaves, 1]
		# Concatenate the routing probabilities with their complements.
		decisions = tf.keras.layers.concatenate([decisions, 1 - decisions], axis=2)  # [batch_size, num_leaves, 2]

		mu = tf.ones([batch_size, 1, 1])

		begin_idx = 1
		end_idx = 2
		# Traverse the tree in breadth-first order.
		for level in range(self.depth):
			mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
			mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
			level_decisions = decisions[:, begin_idx:end_idx, :]  # [batch_size, 2 ** level, 2]
			mu = mu * level_decisions  # [batch_size, 2**level, 2]
			begin_idx = end_idx
			end_idx = begin_idx + 2 ** (level + 1)

		mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
		probabilities = tf.keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
		outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
		return outputs

class NeuralDecisionForest(Model):
	def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
		super(NeuralDecisionForest, self).__init__()
		self.ensemble = []
		self.num_classes = num_classes
		# Initialize the ensemble by adding NeuralDecisionTree instances.
		# Each tree will have its own randomly selected input features to use.
		for _ in range(num_trees):
			self.ensemble.append(NeuralDecisionTree(depth, num_features, used_features_rate, num_classes))

	def call(self, inputs, training=None, mask=None):
		# Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
		batch_size = tf.shape(inputs)[0]
		outputs = tf.zeros([batch_size, self.num_classes])

		# Aggregate the outputs of trees in the ensemble.
		for tree in self.ensemble:
			outputs += tree(inputs)
		# Divide the outputs by the ensemble size to get the average.
		outputs /= len(self.ensemble)
		return outputs