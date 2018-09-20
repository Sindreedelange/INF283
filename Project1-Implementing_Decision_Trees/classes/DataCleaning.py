# coding: utf-8

import pandas as pd
import numpy as np

class DataCleaning(object):

    def __init__(self):
        pass

    def removeQmarksDf(self, dataframe):
        """ Removes rows containing '?' from a dataframe"""
        return dataframe[~(dataframe == '?').any(axis=1)]  

    def factorizeDf(self, dataframe):
        """ Factorizes inputed dataframe, so that its data is categorical, discrete values"""
        data_factorized = dataframe
        for column in dataframe:
            data_factorized[column], unique_data = pd.factorize(dataframe[column])
        return data_factorized