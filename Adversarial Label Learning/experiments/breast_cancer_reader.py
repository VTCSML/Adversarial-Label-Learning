import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader

def load_and_process_data(path):

    """Loads breast cancer data from `path`"""
    df = pd.read_csv(path, header=None)
    #Remove the first column of the data (the ID)
    df.drop([0], axis=1, inplace=True)
    #replace labels 'B' with 0 and 'M' with 1
    df[1] = df[1].replace({'B': 0, 'M': 1})
    data_matrix = df.values
    #Split the data into 70% training and 30% test set
    data_labels = data_matrix[:, :1].ravel() 
    data_matrix = data_matrix[:, 1:]
    train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels, test_size=0.3, shuffle=True, stratify=data_labels)

    #Normalize the features of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    assert train_labels.size == train_data.shape[0]
    assert test_labels.size == test_data.shape[0]

    data = {}

    val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels, test_size=0.4285, shuffle=True, stratify=train_labels)

    data['training_data'] = weak_supervision_data, weak_supervision_labels
    data['validation_data'] = val_data, val_labels
    data['test_data'] = test_data, test_labels

    return data


def run_experiment(run, save):

    """
    :param run: method that runs real experiment given data
    :type: function
    :param save: method that saves experiment results to JSON file
    :type: function
    :return: none
    """

    #for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
    views = {0:0, 1:10, 2:20}
    datapath = 'datasets/breast-cancer/wdbc.data'
    savepath = 'results/json/breast_cancer.json'
    default_reader.run_experiment(run, save, views, datapath, load_and_process_data, savepath)


def run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """

    #for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
    views = {0:0, 1:10, 2:20}
    path = 'results/json/bc_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/breast-cancer/wdbc.data', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)