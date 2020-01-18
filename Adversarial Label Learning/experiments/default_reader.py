import numpy as np
import json

def create_weak_signal_view(path, views, load_and_process_data):
    """
    :param path: relative path to the dataset
    :type: string
    :param views: dictionary containing the index of the weak signals where the keys are numbered from 0
    :type: dict
    :param load_and_process_data: method that loads the dataset and process it into a table form
    :type: function
    :return: tuple of data and weak signal data
    :return type: tuple
    """

    data = load_and_process_data(path)

    train_data, train_labels = data['training_data']
    val_data, val_labels = data['validation_data']
    test_data, test_labels = data['test_data']

    weak_signal_train_data = []
    weak_signal_val_data = []
    weak_signal_test_data = []

    for i in range(len(views)):
        f = views[i]

        weak_signal_train_data.append(train_data[:, f:f+1])
        weak_signal_val_data.append(val_data[:, f:f+1])
        weak_signal_test_data.append(test_data[:, f:f+1])

    weak_signal_data = [weak_signal_train_data, weak_signal_val_data, weak_signal_test_data]

    return data, weak_signal_data


def run_experiment(run, save, views, datapath, load_and_process_data, savepath):

    """
    :param run: method that runs real experiment given data
    :type: function
    :param save: method that saves experiment results to JSON file
    :type: function
    :param views: dictionary of indices for the weak signals
    :type: dict
    :param datapath: relative path to the dataset
    :type: string
    :param load_and_process_data: default method to load and process the given dataset
    :type: function
    :param savepath: relative path to save the results of the experiments
    :type: string
    :return: none
    """

    # set up your variables
    total_weak_signals = 3
    num_experiments = 1

    for i in range(num_experiments):

    	data, weak_signal_data = create_weak_signal_view(datapath, views, load_and_process_data)
    	for num_weak_signal in range(1, total_weak_signals + 1):
    	    adversarial_model, weak_model = run(data, weak_signal_data, num_weak_signal)
    	    print("Saving results to file...")
    	    # save(adversarial_model, weak_model, savepath)


def run_dep_error_exp(run, data_and_weak_signal_data, path):

	"""
	:param run: method that runs real experiment given data
	:type: function
	:return: none
	:param data_and_weak_signal_data: tuple of data and weak signal data
	:type: tuple
	:param path: relative path to save the bounds experiment results
	:type: string
	"""

	# set up your variables
	num_experiments = 10

	all_accuracy = []
	baseline_accuracy = []
	ge_accuracy = []
	weak_signal_accuracy = []

	data, weak_signal_data = data_and_weak_signal_data

	for num_weak_signal in range(num_experiments):
	    output = run(data, weak_signal_data, num_weak_signal + 1)
	    all_accuracy.append(output['ALL'])
	    baseline_accuracy.append(output['AVG'])
	    ge_accuracy.append(output['GE'])
	    weak_signal_accuracy.append(output['WS'])

	print("Saving results to file...")
	filename = path

	output = {}
	output ['ALL'] = all_accuracy
	output['GE'] = ge_accuracy
	output['AVG'] = baseline_accuracy
	output ['WS'] = weak_signal_accuracy

	with open(filename, 'w') as file:
	    json.dump(output, file, indent=4, separators=(',', ':'))
	file.close()



def run_bounds_experiment(run, data_and_weak_signal_data, path):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    :param data_and_weak_signal_data: tuple of data and weak signal data
    :type: tuple
    :param path: relative path to save the bounds experiment results
    :type: string
    """

    data, weak_signal_data = data_and_weak_signal_data

    # set up your variables
    num_weak_signal = 3
    num_experiments = 100
    errors = []
    accuracies = []
    ineq_constraints = []
    weak_signal_ub = []
    weak_test_accuracy = []

    bounds = np.linspace(0, 1, num_experiments)

    for i in range(num_experiments):
        output = run(data, weak_signal_data, num_weak_signal, bounds[i])
        errors.append(output['error_bound'])
        accuracies.append(output['test_accuracy'])
        ineq_constraints.append(output['ineq_constraint'])
        weak_signal_ub.append(output['weak_signal_ub'])
        weak_test_accuracy.append(output['weak_test_accuracy'])

    print("Saving results to file...")
    filename = path

    output = {}
    output ['Error bound'] = errors
    output['Accuracy'] = accuracies
    output['Ineq constraint'] = ineq_constraints
    output ['Weak_signal_ub'] = weak_signal_ub
    output['Weak_test_accuracy'] = weak_test_accuracy
    with open(filename, 'w') as file:
        json.dump(output, file, indent=4, separators=(',', ':'))
    file.close()
