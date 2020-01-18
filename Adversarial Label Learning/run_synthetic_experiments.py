"""Test class for principal components analysis and Gaussian mixture modeling."""
import numpy as np
from train_classifier import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utilities import *


def linear_predict(data, model):
    """
    Predicts a multi-class output based on scores from linear combinations of features.

    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size
                    (d, num_classes) ndarray
    :type model: dict
    :return: length n vector of class predictions
    :rtype: array
    """
    # TODO fill in your code to predict the class by finding the highest scoring linear combination of features
    W=model['weights']
    Y=W.T.dot(data)
    return np.argmax(Y,axis=0)



def generate_data(num_points, num_dim):
    """
    Generates synthetic data based input dimensions

    :param num_points: number of data points to be generated
    :type num_points: int
    :param num_dim: number of features
    :type num_dim: int
    :return: dictionary containing tuples of training, validation and test data
    :rtype: dict
    """

    num_classes = 2
    seed = np.random.randint(0, num_points)
    np.random.seed(seed)
    # Python is not always consistent across machines with preserving seeded random behavior,
    # so if your histogram shows major class imbalance, change this seed to get better balance

    data = np.random.randn(num_dim, num_points)
    true_model = {'weights': np.random.randn(num_dim, num_classes)}

    labels = linear_predict(data, true_model)

    #Divide the data into training, validation and testing data
    num_train = int(num_points * 0.05)
    num_validation = int(num_points * 0.7)

    validation_ind = num_train + num_validation

    #Divide the data into training, validation and test data
    train_data = data[:, :num_train]
    validation_data = data[:, num_train : validation_ind]
    test_data = data[:, validation_ind:]

    #Divide the labels into training, validation and test_labels
    train_labels = labels[:num_train]
    validation_labels = labels[num_train : validation_ind]
    test_labels = labels[validation_ind:]

    data = {}

    data['training_data'] = (train_data, train_labels)
    data['validation_data'] = (validation_data, validation_labels)
    data['test_data'] = (test_data, test_labels)

    return data



def train_weak_signals(data, n_weak_functions):
    """
    Trains weak signals

    :param data: dictionary of training, cross validation and test data
    :type data: dict
    :param n_weak_functions: number of weak signals to be trained
    :type n_weak_functions: int
    :return: dictionary containing of models, probabilities and error bounds of weak signals
    :rtype: dict
    """

    train_data = data['training_data'][0]
    train_labels = data['training_data'][1]
    validation_data = data['validation_data'][0]
    validation_labels = data['validation_data'][1]

    #Divide the training data evenly for the weak functions
    #Training data is transposed so it can be used in scikit learn module

    num_train = train_data.shape[1]
    train_data_list = []
    beg = 0
    n = num_train
    num_train = int(num_train / n_weak_functions)
    for i in range(1, n_weak_functions + 1):
        end = num_train*i
        if n - end < num_train:
            end = n
        func_tr_data = train_data[:, beg:end].T
        func_tr_labels = train_labels[beg:end]
        train_data_list.append([func_tr_data,func_tr_labels])
        beg = num_train*i

    w_sig_probabilities = []
    stats = np.zeros(n_weak_functions)
    weak_function_models = []
    for i in range(n_weak_functions):
        # fit model
        model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        try:
            model.fit(train_data_list[i][0], train_data_list[i][1])
        except:
            print("The experiment failed because the data contains only one class, try again..")
            sys.exit(0)
        weak_function_models.append(model)

        # evaluate probability of P(X=1)
        probability = model.predict_proba(validation_data.T)[:, 1]
        score = validation_labels * (1 - probability) + (1 - validation_labels) * probability
        stats[i] = np.sum(score) / score.size
        w_sig_probabilities.append(probability)

    model = {}
    model['models'] = weak_function_models
    model['probabilities'] = np.array(w_sig_probabilities)
    model['error_bounds'] = stats

    return model



def reportResults(num_weak_signal):
    """
    Print results of data and save to file

    :param num_weak_signal: number of weak signals
    :type num_weak_signal: int
    """

    # set up your variables
    n_weak_functions = num_weak_signal
    n_learnable_classifiers = 1
    num_data_points = 5000
    num_features = 20
    data = generate_data(num_data_points, num_features)
    w_model = train_weak_signals(data, n_weak_functions)
    validation_data = data['validation_data'][0]
    validation_labels = data['validation_data'][1]
    test_data = data['test_data'][0]
    test_labels = data['test_data'][1]

    weak_signal_ub = w_model['error_bounds']
    models = w_model['models']
    print('Weak signal error bounds', weak_signal_ub)
    weak_signal_probabilities = w_model['probabilities']

    # calculate baseline
    baselines = runBaselineTests(validation_data, weak_signal_probabilities)
    weights = np.array([np.zeros(num_features) for i in range(n_learnable_classifiers)])

    print("Running tests...")
    optimized_weights, ineq_constraints = train_all(validation_data, weights, weak_signal_probabilities, weak_signal_ub)

    # calculate validation results
    learned_probabilities = probability(validation_data, optimized_weights)
    validation_accuracy = getModelAccuracy(learned_probabilities, validation_labels)

    # calculate test results
    learned_probabilities = probability(test_data, optimized_weights)
    test_accuracy = getModelAccuracy(learned_probabilities, test_labels)

    # calculate weak signal results
    weak_val_accuracy = getWeakSignalAccuracy(validation_data, validation_labels, models)
    weak_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, models)

    adversarial_model = {}
    adversarial_model['validation_accuracy'] = validation_accuracy
    adversarial_model['test_accuracy'] = test_accuracy

    weak_model = {}
    weak_model['num_weak_signal'] = n_weak_functions
    weak_model['validation_accuracy'] = weak_val_accuracy
    weak_model['test_accuracy'] = weak_test_accuracy

    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("We trained %d learnable classifiers with %d weak signals" %(n_learnable_classifiers, n_weak_functions))
    print("The accuracy of learned model on the validation data is", validation_accuracy)
    print("The accuracy of the model on the test data is", test_accuracy,"\n")
    print("The accuracy of weak signal(s) on the validation data is ", weak_val_accuracy)
    print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("Running tests on the baselines...")
    b_validation_accuracy = getWeakSignalAccuracy(validation_data, validation_labels, baselines)
    b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
    print("The accuracy of the baseline models on validation data is", b_validation_accuracy)
    print("The accuracy of the baseline models on test data is", b_test_accuracy)
    print("")
    weak_model['baseline_val_accuracy'] = b_validation_accuracy
    weak_model['baseline_test_accuracy'] = b_test_accuracy
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return adversarial_model, weak_model


def test_all():
    """
    Tests the minimize agreement rate method.
    :return: None
    """
     # set up your variables
    num_weak_signal = 20

    adversarial_model, weak_model = reportResults(num_weak_signal)
    print("Saving results to file...")
    saveToFile(adversarial_model, weak_model,'synthetic-experiments.json')

if __name__ == '__main__':
    test_all()
