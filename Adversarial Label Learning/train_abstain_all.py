import numpy as np
import sys

def error_calculation(y, q):
    return np.dot(1 - q, y) + np.dot(q, 1 - y)


def inequality_constraint(y, weak_signal_probabilities, weak_signal_ub):
    """
    Computes the gradient of lagrangian inequality penalty parameters

    :param y: vector of estimated labels for the data
    :type y: array
    :param weak_signal_probabilities: soft or hard weak estimates for the data labels
    :type weak_signal_probabilities: ndarray, size no of weak signals x no of datapoints
    :param weak_signal_ub: upper bounds error rates for the weak signals
    :type weak_signal_ub: array
    :return: vector of length gamma containing the gradient of gamma
    :rtype: array
    """
    _, n = weak_signal_probabilities.shape
    mask = weak_signal_probabilities >=0

    weak_term = y * (1 - weak_signal_probabilities) + weak_signal_probabilities * (1-y)
    weak_term = weak_term * mask

    with np.errstate(divide='ignore', invalid='ignore'):
        weak_term = np.sum(weak_term, axis=1) / np.sum(mask, axis=1)
        weak_term = np.nan_to_num(weak_term)

    ineq_constraint = weak_term - weak_signal_ub
    return ineq_constraint


def objective_function(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma):
    """
    Computes the value of the objective function

    :param y: vector of estimated labels for the data
    :type y: array, size n
    :param learnable_probabilities: estimated probabilities for the classifier
    :type learnable_probabilities: array, size n
    :param weak_signal_probabilities: soft or hard weak estimates for the data labels
    :type weak_signal_probabilities: ndarray, size no of weak signals x no of datapoints
    :param weak_signal_ub: upper bounds error rates for the weak signals
    :type weak_signal_ub: array
    :param rho: Scalar tuning hyperparameter
    :type rho: float
    :param gamma: vector of lagrangian inequality penalty parameters
    :type gamma: array
    :return: scalar value of objective function
    :rtype: float
    """

    n = learnable_probabilities.size
    objective = error_calculation(y, learnable_probabilities)
    objective = np.sum(objective) / n

    ineq_constraint = inequality_constraint(y, weak_signal_probabilities, weak_signal_ub)
    gamma_term = np.dot(gamma.T, ineq_constraint)

    ineq_constraint = ineq_constraint.clip(min=0)
    ineq_augmented_term = (rho/2) * ineq_constraint.T.dot(ineq_constraint)

    return objective + gamma_term - ineq_augmented_term


def logistic(x):
    """
    Squashing function that returns the squashed value and a vector of the
                derivatives of the squashing operation.

    :param x: ndarray of inputs to be squashed
    :type x: ndarray
    :return: tuple of (1) squashed inputs and (2) the gradients of the
                squashing function, both ndarrays of the same shape as x
    :rtype: tuple
    """
    y = 1 / (1 + np.exp(-x))
    grad = y * (1 - y)

    return y, grad


def probability(data, weights):
    """
    Computes the probabilities of the data for the classifier

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param weights: vectors of weights for the classifier
    :type weights: ndarray, size d
    :return: size n prababilities for the classifier
    :rtype: array
    """

    try:
        y = weights.dot(data)
    except:
        y = data.dot(weights)

    probs, _ = logistic(y)
    return probs


def weight_gradient(data, weights):
    """
    Computes the gradient the probabilities wrt to the weights

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param weights: size  d containing vector of weights for each learnable function
    :type weights: array
    :return: ndarray of size (n_of_features, n) gradients for probability wrt to weight
    :rtype: ndarray
    """

    try:
        y = weights.dot(data)
    except:
        y = data.dot(weights)

    _, grad = logistic(y)
    grad = data * grad
    return grad


def y_gradient(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma):
    """
    Computes the gradient y

    See description in objective function for the variables
    :return: vector of length y containing the gradient of y
    :rtype: array
    """
    n = learnable_probabilities.size
    learnable_term = 1 - (2 * learnable_probabilities)
    learnable_term = np.sum(learnable_term, axis=0) / n

    mask = weak_signal_probabilities >=0
    ls_term = 1 - (2 * weak_signal_probabilities)
    ls_term = ls_term * mask

    gamma_term = np.dot(gamma.T, ls_term) / n

    ineq_constraint = inequality_constraint(y, weak_signal_probabilities, weak_signal_ub)
    ineq_constraint = ineq_constraint.clip(min=0)
    ineq_augmented_term = rho * np.dot(ineq_constraint.T, ls_term)

    return learnable_term + gamma_term - ineq_augmented_term


def train_all(data, weak_signal_probabilities, weak_signal_ub, max_iter=5000):

    """
    Trains a logistic regression classifier
    The model for the classifier can be replaced

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    See description in objective function for other variables
    :return: tuple of ndarray containing optimized weights for the classifier and vector of inequality constraints
    :rtype: tuple
    """

    d, n = data.shape
    weights = np.zeros(d)
    learnable_probabilities = probability(data, weights)

    # initialize algorithm variables
    y = 0.5 * np.ones(n)
    gamma = np.zeros(weak_signal_probabilities.shape[0])
    one_vec = np.ones(n)
    rho = 2.5
    lr = 0.0001

    t = 0
    converged = False
    while not converged and t < max_iter:
        rate = 1 / (1 + t)

        # update y
        old_y = y
        y_grad = y_gradient(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma)
        y = y + rate * y_grad
        # projection step: clip y to [0, 1]
        y = y.clip(min=0, max=1)

        # compute gradient of probabilities
        dl_dp = (1 / n) * (1 - 2 * old_y)

        # update gamma
        old_gamma = gamma
        gamma_grad = inequality_constraint(old_y, weak_signal_probabilities, weak_signal_ub)
        gamma = gamma - rho * gamma_grad
        gamma = gamma.clip(max=0)

        weights_gradient = []
        # compute gradient of probabilities wrt weights
        dp_dw = weight_gradient(data, weights)
        # update weights
        old_weights = weights.copy()
        weights_gradient.append(dp_dw.dot(dl_dp))

        # update weights of the learnable functions
        weights = weights - lr * np.array(weights_gradient)
        conv_weights = np.linalg.norm(weights - old_weights)
        conv_y = np.linalg.norm(y - old_y)

        # check that inequality constraints are satisfied
        ineq_constraint = inequality_constraint(y, weak_signal_probabilities, weak_signal_ub)
        ineq_infeas = np.linalg.norm(ineq_constraint.clip(min=0))

        converged = np.isclose(0, conv_y, atol=1e-6) and np.isclose(0, ineq_infeas, atol=1e-6) and np.isclose(0, conv_weights, atol=1e-5)
        converged = False

        if t % 1000 == 0:
            lagrangian_obj = objective_function(y, learnable_probabilities, weak_signal_probabilities, weak_signal_ub, rho, gamma) # might be slow
            objective = np.dot(learnable_probabilities, 1 - y) + np.dot(1 - learnable_probabilities, y)
            objective = np.sum(objective) / n
            print("Iter %d. Weights Infeas: %f, Y_Infeas: %f, Ineq Infeas: %f, lagrangian: %f, obj: %f" % (t, np.sum(conv_weights), conv_y,
                                                ineq_infeas, lagrangian_obj, objective))
        learnable_probabilities = probability(data, weights)
        t += 1

    # print("Inequality constraints", ineq_constraint)
    # print("Weak signal upper bounds: ", weak_signal_ub)
    learned_probabilities = probability(data, weights)

    return learned_probabilities, weights
