import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json, sys

def getModelAccuracy(probabilities, labels, probability_score=None):
    """
    Print accuracy of the learned model

    :param probabilities: vector of probabilities of the learned classifier
    :type probabilities: array
    :param labels: array of labels for the data
    :type labels: array
    :param probability_score: indicator of which accuracy score to return
    :type probability_score: any
    :return: accuracy of the trained model
    :rtype: float
    """
    probabilities = probabilities.ravel()
    if probability_score is not None:
        predictions = labels * (1 - probabilities) + (1 - labels) * probabilities
        score = np.sum(predictions) / predictions.size
        score = 1 - score
    else:
        predictions = np.zeros(probabilities.size)
        predictions[probabilities > 0.5] =1
        score = accuracy_score(labels, predictions)

    return score


def getWeakSignalAccuracy(data, labels, models, probability_score=None):
    """
    Print accuracy of weak signals on the data

    :param data: ndarray of datapoints
    :type data: ndarray
    :param labels: array of labels for the data
    :type labels: array
    :param models: list containing trained models for the weak signals
    :type models: list
    :return: accuracy of the weak signal model(s)
    :param probability_score: indicator of which accuracy score to return
    :type probability_score: any
    :rtype: list
    """
    stats = []

    for model in models:
        p = model.predict_proba(data.T)[:,1]

        if probability_score is not None:
            predictions = labels * (1 - p) + (1 - labels) * p
            score = np.sum(predictions) / predictions.size
            score = 1 - score
            stats.append(score)
        else:
            predictions = np.zeros(p.size)
            predictions[p > 0.5] =1
            score = accuracy_score(labels, predictions)
            stats.append(score)

    return stats


def runBaselineTests(data, weak_signal_probabilities):
    """
    Run baseline tests using each of the weak signals and average of all weak signals

    :param data: ndarray of datapoints
    :type data: ndarray
    :param weak_signal_probabilities: matrix of weak signal probabilities to be used as labels
    :type weak_signal_probabilities: ndarray
    :return: lists containing the baseline models
    :rtype: list
    """

    baselines = []
    average_weak_labels = np.mean(weak_signal_probabilities, axis=0)
    average_weak_labels[average_weak_labels > 0.5] = 1
    average_weak_labels[average_weak_labels <= 0.5] = 0

    model = LogisticRegression(solver = "lbfgs", max_iter= 1000)
    try:
        model.fit(data.T, average_weak_labels)
    except:
        print("The mean of the baseline labels is %f" %np.mean(average_weak_labels))
        sys.exit(1)

    baselines.append(model)

    return baselines


def saveToFile(adversarial_model, weak_signal_model, filename):

    """
    Appends the test results to a file

    :param adversarial_model: dictionary containing results of learned model
    :type adversarial_model: dict
    :param num_features: dictionary containing results of the weak signals
    :type weak_signal_model: dict
    :param filename: name of the file to save reults
    :type filename: string
    :return: None
    :rtype: None
    """

    output = {}
    output ['Adversarial model'] = adversarial_model
    output ['Weak model'] = weak_signal_model
    with open(filename, 'a') as file:
        json.dump(output, file, indent=4, separators=(',', ':'))
    file.close()
