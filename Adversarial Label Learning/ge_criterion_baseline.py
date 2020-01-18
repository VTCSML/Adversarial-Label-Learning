import numpy as np
from train_classifier import logistic
from scipy.optimize import minimize, check_grad
from sklearn.metrics import accuracy_score

def compute_reference_distribution(labels, weak_signal):
	"""
	Computes the score value of the reference expectation

	:param labels: size n labels for each instance in the dataset
	:type labels: array
	:param weak_signal: weak signal trained using one dimensional feature
	:type  weak_signal: array
	:return: tuple containing scalar values of positive and negative reference probability distribution
	:rtype: float
	"""
	threshold = 0.5
	positive_index = np.where(weak_signal >= threshold)
	negative_index = np.where(weak_signal < threshold)
	pos_feature_labels = labels[positive_index]
	neg_feature_labels = labels[negative_index]

	try:
	    with np.errstate(all='ignore'):
	    	reference_pos_probability = np.sum(pos_feature_labels) / pos_feature_labels.size
	    	reference_neg_probability = np.sum(neg_feature_labels) / neg_feature_labels.size
	except:
		reference_pos_probability = np.nan_to_num(np.sum(pos_feature_labels) / pos_feature_labels.size) + 0
		reference_neg_probability = np.nan_to_num(np.sum(neg_feature_labels) / neg_feature_labels.size) + 0

	return reference_pos_probability, reference_neg_probability


def ge_criterion_train(data, labels, weak_signal_probabilities, num_weak_signals, check_gradient=False):
	"""
	Trains generalized expectation criteria

	:param data: size (n, d) ndarray containing n examples described by d features each
	:type data: ndarray
	:param labels: length n array of the integer class labels
	:type labels: array
	:param weak_signal_probabilities: size num_weak_signals x n of the weak signal probabilities
	:type weak_signal_probabilities: ndarray
	:param num_weak_signals: the number of weak signal to be used in training
	:type num_weak_signals: integer
	:return: the learned model
	:rtype: array
	"""

	n, d = data.shape
	weights = np.random.rand(d)

	def compute_empirical_distribution(est_probability, weak_signal):
		"""
		Computes the score value of the emperical distribution

		:param est_probability: size n estimated probabtilities for the instances
		:type labels: array
		:param weak_signal: weak signal trained using one dimensional feature
		:type  weak_signal: array
		:return: (tuple of scalar values of the empirical distribution, tuple of index of instances)
		:rtype: tuple
		"""
		threshold = 0.5
		positive_index = np.where(weak_signal >= threshold)
		negative_index = np.where(weak_signal < threshold)
		pos_feature_labels = est_probability[positive_index]
		neg_feature_labels = est_probability[negative_index]

		try:
		    with np.errstate(all='ignore'):
		    	empirical_pos_probability = np.sum(pos_feature_labels) / pos_feature_labels.size
	    		empirical_neg_probability = np.sum(neg_feature_labels) / neg_feature_labels.size
		except:
			empirical_pos_probability = np.nan_to_num(np.sum(pos_feature_labels) / pos_feature_labels.size) + 0
			empirical_neg_probability = np.nan_to_num(np.sum(neg_feature_labels) / neg_feature_labels.size) + 0

		empirical_probability = empirical_pos_probability, empirical_neg_probability
		instances_index = positive_index, negative_index
		return empirical_probability, instances_index

	def train_ge_criteria(new_weights):
		"""
		This internal function returns the objective value of ge criteria

		:param new_weights: weights to use for computing multinomial logistic regression
		:type new_weights: ndarray
		:return: tuple containing (objective, gradient)
		:rtype: (float, array)
		"""

		obj = 0
		score = data.dot(new_weights)
		probs, grad = logistic(score)
		gradient = 0
		# Code to compute the objective function
		for i in range(num_weak_signals):
			weak_signal = weak_signal_probabilities[i]
			reference_probs = compute_reference_distribution(labels, weak_signal)
			empirical_probs, index = compute_empirical_distribution(probs, weak_signal)

			# empirical computations
			pos_empirical_probs, neg_empirical_probs = empirical_probs
			pos_index, neg_index = index

			# reference computations
			pos_reference_probs, neg_reference_probs = reference_probs

			try:
				with np.errstate(all='ignore'):
					# compute objective for positive probabilities
					obj += pos_reference_probs * np.log(pos_reference_probs / pos_empirical_probs)
					gradient += (pos_reference_probs / pos_empirical_probs) * data[pos_index].T.dot(grad[pos_index]) / grad[pos_index].size

					# compute objective for negative probabilities
					obj += neg_reference_probs * np.log(neg_reference_probs / neg_empirical_probs)
					gradient += (neg_reference_probs / neg_empirical_probs) * data[neg_index].T.dot(grad[neg_index]) / grad[neg_index].size
			except:
				# compute objective for positive probabilities
				obj += np.nan_to_num(pos_reference_probs * np.log(pos_reference_probs / pos_empirical_probs))
				gradient += np.nan_to_num((pos_reference_probs / pos_empirical_probs) * data[pos_index].T.dot(grad[pos_index]) / grad[pos_index].size)

				# compute objective for negative probabilities
				obj += np.nan_to_num(neg_reference_probs * np.log(neg_reference_probs / neg_empirical_probs))
				gradient += np.nan_to_num((neg_reference_probs / neg_empirical_probs) * data[neg_index].T.dot(grad[neg_index]) / grad[neg_index].size)

		objective = obj + (0.5 * np.sum(new_weights**2))
		gradient = new_weights - gradient

		return objective, gradient

	if check_gradient:
	    grad_error = check_grad(lambda w: train_ge_criteria(w)[0], lambda w: train_ge_criteria(w)[1].ravel(), weights)
	    print("Provided gradient differed from numerical approximation by %e (should be below 1e-3)" % grad_error)

	# pass the internal objective function into the optimizer
	res = minimize(lambda w: train_ge_criteria(w)[0], jac=lambda w: train_ge_criteria(w)[1].ravel(), x0=weights)
	weights = res.x

	return weights

"""
data = np.random.randn(100, 50)
labels = np.random.randint(2, size=100)
weak_signal_probabilities = np.random.rand(3, 100)
num_weak_signals = 1
model = ge_criterion_train(data, labels, weak_signal_probabilities, num_weak_signals, check_gradient=True)
"""
