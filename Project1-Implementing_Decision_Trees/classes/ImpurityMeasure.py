
# coding: utf-8

import numpy as np
import pandas as pd
from math import log2

class ImpurityMeasure(object):  
    
    def __init__(self, purity_measure):
        if purity_measure=="gini" or purity_measure=="entropy":
            self.purity_measure = purity_measure
        else:
            self.purity_measure="entropy"
    
    def randomness_reduction(self, entropy_src, entropy_branch):
        """ Calculates the reduction in randomness, aka Information Gain.
        entropy_src = entropy of the entire system (target variable),
        entropy_branch = entropy for one branch
        Returns the Information Gain - restricted to 3 decimals."""
        return (round(entropy_src - entropy_branch, 3))

    def calc_entropy(self, p):
        """Calculate the entropy for a given fraction"""
        if p!=0:
            return -p * np.log2(p)
        else:
            return 0
    
    def calc_gini(self, p):
        """" Calculate the gini index for a given fraction"""
        if p!=0:
            return p * (1-p)
        else:
            return 0

    def calc_entropy_system(self, target_variable):
        """Calculates the entropy of the (entire) system, i.e. target variable"""
        tot_len = len(target_variable)
        unique, counts = np.unique(target_variable, return_counts=True)
        dic = dict(zip(unique, counts))
        purity = 0
        for key, value in dic.items():
            if self.purity_measure=="entropy":
                purity += self.calc_entropy(value/tot_len)
            else:
                purity += self.calc_gini(value/tot_len)
        return purity

    def calc_entropy_all_branches(self, feature_entr_dict, entropy_src):
        """Calculates the entropy among all the branches
        Expects a dictionary on the format 
            {'column feature': 
                {unique value: [
                    number of this value occured in the set
                    number of values in the set
                    entropy when this value occured in the set
                ], ... }}"""
        column_entropy_dict_full = {}
        for column_feature in feature_entr_dict:
            entropy_all = 0
            # Take each value from each 'unique value', for each 'column feature' from the inputed dictionary
            # and calculates the entropy for each 'unique value', e.g. sunny, rainy, etc. 
            for unique_val in feature_entr_dict[column_feature]:
                # NOTE: As mentioned in PyDoc - assumes this format
                num_val = feature_entr_dict[column_feature][unique_val][0]
                num_tot = feature_entr_dict[column_feature][unique_val][1]
                num_val_entropy = feature_entr_dict[column_feature][unique_val][2]
                entropy_all += (num_val/num_tot)*num_val_entropy
            column_entropy_dict_full[column_feature] = entropy_all
        
        # Calculate the information gain for each column
        information_gain_dict = {}
        for key, value in column_entropy_dict_full.items():
            information_gain_dict[key] = self.randomness_reduction(entropy_src, value)
        return information_gain_dict
    
    def calc_entropy_feature(self, X_y_zip, tot_num_occurences):
        """ Calculates the necessary numbers, to calculate the entropy - store it in a dictionary.
        X_y_zip = a dictionary where each unique value, in each column (from X), is mapped to their respective
        target variable value. 
        tot_num_occurences = number of target points (length of the columns)"""
        # Dict to store the number necessary to calculate the entropy
        columns_entropy = {}
        for feature in X_y_zip:
            # Get unique variables for each key, aka each column (from X)
            list_of_unique_variables = list(set([x[0] for x in X_y_zip[feature].values]))

            val_dict = {}
            for val in list_of_unique_variables:
                # Total number of days for each unique variable (from X)
                num_days_val = len([x[1] for x in X_y_zip[feature] if x[0] == val])
                # Total number of days for each key (assuming it is binary)
                num_days_val_yes = len([x[1] for x in X_y_zip[feature] if x[0] == val and x[1] == 1])
                num_days_val_no = num_days_val - num_days_val_yes
                #Calculate entropy for each unique value
                if self.purity_measure=="entropy":
                    val_entropy = self.calc_entropy(num_days_val_yes/num_days_val) + self.calc_entropy(num_days_val_no/num_days_val)
                else:
                    val_entropy = self.calc_gini(num_days_val_yes/num_days_val) + self.calc_gini(num_days_val_no/num_days_val)
                # Make a list with relevant data for each unique value
                val_list = [num_days_val, tot_num_occurences, val_entropy]
                # Append that list to a dictionary, where the unique value is key
                val_dict[val] = val_list
            # Append dictionaries for unique values, to their respectively feature/column
            columns_entropy[feature] = val_dict
        return columns_entropy
    
    
    def calc_entropy_dataset(self, X, y):
        """Calculates the entropy of an entire dataset, represented by 
        X and y, training- and target variables, respectively"""
        # Number of datapoints in the set
        tot_num_occurences = len(y)
        # Calculate the impurity of the target variable
        entropy_system = self.calc_entropy_system(y)
        # Small hack because the system was initially made for full Dataframe input
        data = pd.concat([X, y], axis=1)
        X_y_zip = {}
        for columns in X:
            # Map each value in each column in X to their respective "outcome"/target variable (y)
            X_y_zip[columns] = data[[columns, y.name]].apply(tuple, axis=1)
        # Calculate the entropy for each feature - using the just made dictionary
        each_feature_w_entropy = self.calc_entropy_feature(X_y_zip, tot_num_occurences)

        return self.calc_entropy_all_branches(each_feature_w_entropy, entropy_system)
    
    def getLargestInformationGain(self, X, y):
        """Gets the largest IG for any given dataset (that is Pandas DataFrame)"""
        ig_dict = self.calc_entropy_dataset(X, y)
        return (max(ig_dict, key=ig_dict.get))

