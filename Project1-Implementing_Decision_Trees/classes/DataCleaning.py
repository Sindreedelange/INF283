# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCleaning(object):

    def __init__(self):
        pass

    def removeQmarksDf(self, dataframe):
        """ Removes rows containing '?' from a dataframe"""
        return dataframe[~(dataframe == '?').any(axis=1)]  

    def factorizeDf(self, dataframe):
        """ Factorizes inputed dataframe, so that its data is categorical, discrete values"""
        pd.options.mode.chained_assignment = None
        data_factorized = dataframe
        for column in dataframe:
            data_factorized[column], unique_data = pd.factorize(dataframe[column])
        return data_factorized

    def train_test_prune(self, X, y, test, prune):
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test, random_state=42, straify=y)
        X_train, X_prune, y_train, y_prune = train_test_split(X_train, y_train, test_size=prune, random_state=42, stratify=y_train)
        return X_train, X_prune, X_test, y_train, y_prune, y_test 