import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from experiments import default_reader

def load_and_process_data(path):

    """Loads text classification data from `path`"""
    df = pd.read_csv(path, header=None)
    #Use 'NB-Wait' and "'NB-No Block'" labels only in the data
    mask_no_block = df[21] == "'NB-No Block'"
    mask_wait = df[21] == 'NB-Wait'
    df = df[mask_no_block | mask_wait].copy(deep=True)

    # remove rows with missing values
    df = df[df[13] != '?']

    #replace node status feature 'NB', 'B' and 'P NB' with 1, 2, 3
    df[19] = df[19].replace(['NB', 'B', "'P NB'"], [1, 2, 3])

    #replace labels "'NB-No Block'" with 0 and 'NB-Wait' with 1
    df[21] = df[21].replace({"'NB-No Block'": 0, 'NB-Wait': 1})
    data_matrix = df.values

    #Split the data into 70% training and 30% test set
    data_labels = data_matrix[:,-1:].ravel() 
    data_matrix = data_matrix[:,:-1]
    train_data, test_data, train_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)

    #Normalize the features of the data
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    assert train_labels.size == train_data.shape[0]
    assert test_labels.size == test_data.shape[0]

    data = {}

    val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_data, train_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_labels)

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

    #for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals
    views = {0:1, 1:2, 2:20}
    datapath = 'datasets/obs-network/obs_network.data'
    savepath = 'results/json/obs_network.json'
    default_reader.run_experiment(run, save, views, datapath, load_and_process_data, savepath)


def run_bounds_experiment(run):

    """
    :param run: method that runs real experiment given data
    :type: function
    :return: none
    """
    #for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals
    views = {0:1, 1:2, 2:20}
    path = 'results/json/obs_bounds.json'
    data_and_weak_signal_data = default_reader.create_weak_signal_view('datasets/breast-cancer/wdbc.data', views, load_and_process_data)
    default_reader.run_bounds_experiment(run, data_and_weak_signal_data, path)