import numpy as np
from train_classifier import *
from ge_criterion_baseline import *
from utilities import saveToFile, runBaselineTests, getModelAccuracy, getWeakSignalAccuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from experiments import breast_cancer_reader, cardio_reader, obs_network_reader


def train_weak_signals(data, weak_signal_data, num_weak_signal):
    """
    Trains different views of weak signals

    :param data: dictionary of training and test data
    :type data: dict
    :param weak_signal_data: data representing the different views for the weak signals
    :type: array
    :param num_weak_signal: number of weak_signals
    type: in
    :return: dictionary containing of models, probabilities and error bounds of weak signals
    :rtype: dict
    """

    train_data, train_labels = data['training_data']
    val_data, val_labels = data['validation_data']
    test_data, test_labels = data['test_data']

    n, d = train_data.shape

    weak_signal_train_data = weak_signal_data[0]
    weak_signal_val_data = weak_signal_data[1]
    weak_signal_test_data = weak_signal_data[2]

    weak_signals = []
    stats = np.zeros(num_weak_signal)
    w_sig_probabilities = []
    w_sig_test_accuracies = []
    weak_val_accuracy = []


    for i in range(num_weak_signal):
        # fit model
        model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
        model.fit(weak_signal_train_data[i], train_labels)
        weak_signals.append(model)

        # evaluate probability of P(X=1)
        probability = model.predict_proba(weak_signal_val_data[i])[:, 1]
        score = val_labels * (1 - probability) + (1 - val_labels) * probability
        stats[i] = np.sum(score) / score.size
        w_sig_probabilities.append(probability)

        # evaluate accuracy for validation data
        weak_val_accuracy.append(accuracy_score(val_labels, np.round(probability)))

        # evaluate accuracy for test data
        test_predictions = model.predict(weak_signal_test_data[i])
        w_sig_test_accuracies.append(accuracy_score(test_labels, test_predictions))


    model = {}
    model['models'] = weak_signals
    model['probabilities'] = np.array(w_sig_probabilities)
    model['error_bounds'] = stats
    model['validation_accuracy'] = weak_val_accuracy
    model['test_accuracy'] = w_sig_test_accuracies

    return model


def run_experiment(data, weak_signal_data, num_weak_signal, constant_bound=False):
    """
    Runs experiment with the given dataset

    :param data: dictionary of validation and test data
    :type data: dict
    :param weak_signal_data: data representing the different views for the weak signals
    :type: array
    :param num_weak_signal: number of weak signals
    :type num_weak_signal: int
    """

    w_model = train_weak_signals(data, weak_signal_data, num_weak_signal)

    training_data = data['training_data'][0].T
    training_labels = data['training_data'][1]
    val_data, val_labels = data['validation_data']
    val_data = val_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = training_data.shape

    weak_signal_ub = w_model['error_bounds']
    # weak_signal_ub = np.ones(w_model['error_bounds'].shape) * 0.3
    models = w_model['models']
    weak_signal_probabilities = w_model['probabilities']

    weights = np.zeros(num_features)

    print("Running tests...")
    if constant_bound:
        optimized_weights, y = train_all(val_data, weights, weak_signal_probabilities, np.zeros(weak_signal_ub.size) + 0.3, max_iter=10000)
    else:
       optimized_weights, y = train_all(val_data, weights, weak_signal_probabilities, weak_signal_ub, max_iter=10000)

    # calculate validation results
    learned_probabilities = probability(val_data, optimized_weights)
    validation_accuracy = getModelAccuracy(learned_probabilities, val_labels)

    # calculate test results
    learned_probabilities = probability(test_data, optimized_weights)
    test_accuracy = getModelAccuracy(learned_probabilities, test_labels)

    # calculate weak signal results
    weak_val_accuracy = w_model['validation_accuracy']
    weak_test_accuracy = w_model['test_accuracy']

    adversarial_model = {}
    adversarial_model['validation_accuracy'] = validation_accuracy
    adversarial_model['test_accuracy'] = test_accuracy

    weak_model = {}
    weak_model['num_weak_signal'] = num_weak_signal
    weak_model['validation_accuracy'] = weak_val_accuracy
    weak_model['test_accuracy'] = weak_test_accuracy

    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("We trained %d learnable classifiers with %d weak signals" %(1, num_weak_signal))
    print("The accuracy of learned model on the validatiion data is", validation_accuracy)
    print("The accuracy of weak signal(s) on the validation data is", weak_val_accuracy)
    print("The accuracy of the model on the test data is", test_accuracy)
    print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # calculate baseline
    print("Running tests on the baselines...")
    baselines = runBaselineTests(val_data, weak_signal_probabilities) #remove the transpose to enable it run
    b_validation_accuracy = getWeakSignalAccuracy(val_data, val_labels, baselines)
    b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
    print("The accuracy of the baseline models on test data is", b_test_accuracy)
    print("")
    weak_model['baseline_val_accuracy'] = b_validation_accuracy
    weak_model['baseline_test_accuracy'] = b_test_accuracy
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # calculate ge criteria
    print("Running tests on ge criteria...")
    model = ge_criterion_train(val_data.T, val_labels, weak_signal_probabilities, num_weak_signal)
    ge_validation_accuracy = accuracy_score(val_labels, np.round(probability(val_data, model)))
    ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
    print("The accuracy of ge criteria on validation data is", ge_validation_accuracy)
    print("The accuracy of ge criteria on test data is", ge_test_accuracy)
    weak_model['gecriteria_val_accuracy'] = ge_validation_accuracy
    weak_model['gecriteria_test_accuracy'] = ge_test_accuracy
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return adversarial_model, weak_model


def bound_experiment(data, weak_signal_data, num_weak_signal, bound):
    """
    Runs experiment with the given dataset

    :param data: dictionary of validation and test data
    :type data: dict
    :param weak_signal_data: data representing the different views for the weak signals
    :type: array
    :param num_weak_signal: number of weak signals
    :type num_weak_signal: int
    :param bound: error bound of the weak signal
    :type bound: int
    :return: outputs from the bound experiments
    :rtype: dict
    """

    w_model = train_weak_signals(data, weak_signal_data, num_weak_signal)

    training_data = data['training_data'][0].T
    training_labels = data['training_data'][1]
    val_data, val_labels = data['validation_data']
    val_data = val_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = training_data.shape

    weak_signal_ub = w_model['error_bounds']
    weak_signal_probabilities = w_model['probabilities']

    weights = np.zeros(num_features)

    print("Running tests...")

    optimized_weights, ineq_constraint = train_all(val_data, weights, weak_signal_probabilities, bound, max_iter=10000)

    # calculate test probabilities
    test_probabilities = probability(test_data, optimized_weights)
    # calculate error bound on test data
    test_probabilities = np.round(test_probabilities)
    error_bound = (test_probabilities.dot(1 - test_labels) + (1 - test_probabilities).dot(test_labels)) / test_labels.size
    test_accuracy = getModelAccuracy(test_probabilities, test_labels)

    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("The error_bound of learned model on the weak signal is", weak_signal_ub)
    print("The test accuracy of the weak signal is", w_model['test_accuracy'])
    print("The error_bound of learned model on the test data is", error_bound[0])
    print("The accuracy of the model on the test data is", test_accuracy)

    output = {}
    output['weak_signal_ub'] = weak_signal_ub[0]
    output['weak_test_accuracy'] = w_model['test_accuracy'][0]
    output['error_bound'] = error_bound[0]
    output['test_accuracy'] = test_accuracy
    output['ineq_constraint'] = ineq_constraint[0]

    return output


def dependent_error_exp(data, weak_signal_data, num_weak_signal):
    """
    Runs experiment with the given dataset

    :param data: dictionary of validation and test data
    :type data: dict
    :param weak_signal_data: data representing the different views for the weak signals
    :type: array
    :param num_weak_signal: number of weak signals
    :type num_weak_signal: int
    :return: test accuracies of ALL and respective baselines
    :rtype: dict
    """

    w_model = train_weak_signals(data, weak_signal_data, num_weak_signal)

    training_data = data['training_data'][0].T
    training_labels = data['training_data'][1]
    val_data, val_labels = data['validation_data']
    val_data = val_data.T
    test_data = data['test_data'][0].T
    test_labels = data['test_data'][1]

    num_features, num_data_points = training_data.shape

    weak_signal_ub = w_model['error_bounds']
    weak_signal_probabilities = w_model['probabilities']
    weak_test_accuracy = w_model['test_accuracy']

    weights = np.zeros(num_features)

    print("Running tests...")

    optimized_weights, ineq_constraint = train_all(val_data, weights, weak_signal_probabilities, weak_signal_ub, max_iter=5000)

    # calculate test probabilities
    test_probabilities = probability(test_data, optimized_weights)
    # calculate test accuracy
    test_accuracy = getModelAccuracy(test_probabilities, test_labels)

    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Experiment %d"%num_weak_signal)
    print("We trained %d learnable classifiers with %d weak signals" %(1, num_weak_signal))
    print("The accuracy of the model on the test data is", test_accuracy)
    print("The accuracy of weak signal(s) on the test data is", weak_test_accuracy)
    print("")

    # calculate ge criteria
    print("Running tests on ge criteria...")
    model = ge_criterion_train(val_data.T, val_labels, weak_signal_probabilities, num_weak_signal)
    ge_test_accuracy = accuracy_score(test_labels, np.round(probability(test_data, model)))
    print("The accuracy of ge criteria on test data is", ge_test_accuracy)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # calculate baseline
    print("Running tests on the baselines...")
    baselines = runBaselineTests(val_data, weak_signal_probabilities)
    b_test_accuracy = getWeakSignalAccuracy(test_data, test_labels, baselines)
    print("The accuracy of the baseline models on test data is", b_test_accuracy)
    print("")

    output = {}
    output['ALL'] = test_accuracy
    output['WS'] = w_model['test_accuracy'][-1]
    output['GE'] = ge_test_accuracy
    output['AVG'] = b_test_accuracy[-1]

    return output


def run_tests():
    """
    Runs experiment.
    :return: None
    """

    print("Running breast cancer experiment...")
    breast_cancer_reader.run_experiment(run_experiment, saveToFile)
    print("Running obs network experiment...")
    obs_network_reader.run_experiment(run_experiment, saveToFile)
    print("Running cardio experiment...")
    cardio_reader.run_experiment(run_experiment, saveToFile)

    # # un-comment to run bounds experimrnt in the paper
    #breast_cancer_reader.run_bounds_experiment(bound_experiment)
    #obs_network_reader.run_bounds_experiment(bound_experiment)

    # # un-comment to run dependency error experiment in the paper
    #print("Running dependent error on cardio experiment...")
    # cardio_reader.run_dep_error_exp(dependent_error_exp)


if __name__ == '__main__':
    run_tests()
