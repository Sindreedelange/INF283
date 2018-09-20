# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCleaning(object):

    def __init__(self):
        pass

    def removeQmarksDf(self, dataframe):
        """ Remove Question Marks from Dataframe 

        Removes rows containing '?' from a dataframe
        
        Args:
            dataframe: pandas dataframe
        
        Returns:
            pandas dataframe
        """
        return dataframe[~(dataframe == '?').any(axis=1)]  

    def factorizeDf(self, dataframe):
        """ Factorizes inputed dataframe
        
        Factorized a dataframe so that its data is categorical, discrete values, only
        
        Args:
            dataframe: pandas dataframe
            
        Returns: 
            pandas dataframe    
        """
        pd.options.mode.chained_assignment = None
        data_factorized = dataframe
        for column in dataframe:
            # unique_data is not in use, for now
            data_factorized[column], unique_data = pd.factorize(dataframe[column])
        return data_factorized

    def train_test_prune(self, X, y, test=0.3, prune=0.3):
        """ Train, test, prune(/validate)
        
        Divides a dataset into train, test, and prune/validate dataset (with their associated labels) 
        - distribution decided by params 'test' and 'prune'

        Args:
            X: pandas dataframe 
            y: pandas series
            test: float
            prune: float
        
        Returns:
            X_train: pandas dataframe
            X_prune: pandas dataframe
            X_test: pandas dataframe
            y_train: pandas series
            y_prune: pandas series
            y_test: pandas series
        """
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test, random_state=42, stratify=y)
        X_train, X_prune, y_train, y_prune = train_test_split(X_train, y_train, test_size=prune, random_state=42, stratify=y_train)
        return X_train, X_prune, X_test, y_train, y_prune, y_test 