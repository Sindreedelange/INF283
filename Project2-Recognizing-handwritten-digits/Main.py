import numpy as np


def load_result_dicts():
    classifiers_dict = np.load("dictionaries/classifiers_dict.npy").item()
    optim_classifier_cnn_params = np.load("dictionaries/optim_classifier_cnn_params.npy").item()
    return classifiers_dict, optim_classifier_cnn_params


class Main:
    classifiers_dict, optim_classifier_cnn_params = load_result_dicts()
    print("Scores for the 4 candidate models: \n", classifiers_dict)
    print("\n Using the following parameters: \n", optim_classifier_cnn_params)





