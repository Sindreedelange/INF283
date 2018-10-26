import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# Works when Keras is activated in the environment
from keras.utils import np_utils


class Data(object):

    def __init__(self):
        self.random_state = 42
        np.random.seed(self.random_state)

    def label_to_categorical(self, y_train, y_val, y_test):
        y_train_cat = np_utils.to_categorical(y_train)
        y_val_cat = np_utils.to_categorical(y_val)
        y_test_cat = np_utils.to_categorical(y_test)
        return y_train_cat, y_val_cat, y_test_cat

    def print_data_shapes_val(self, X_train, X_test, y_train, y_test, y_val = None, X_val = None):
        print("X_train: ", X_train.shape)
        if X_val is not None:
            print("X_val: ", X_val.shape)
        print("X_test: ", X_test.shape)
        print("y_train: ", y_train.shape)
        if y_val is not None:
            print("y_val: ", y_val.shape)
        print("y_test: ", y_test.shape)

    def get_data(self, one_hot_enc=True, reshape=False, train_val_test=False):
        DATA_PATH = "../data/"
        file_list = os.listdir(DATA_PATH)
        images_path = file_list[0]
        labels_path = file_list[1]
        images_path_full = os.path.join(DATA_PATH + images_path)
        labels_path_full = os.path.join(DATA_PATH + labels_path)
        X = pd.read_csv(images_path_full)
        if reshape:
            X = X.values.astype('float32')
            # Normalize data
            X = X / 255
            # Reshape for cnn
            X = X.reshape([-1, 28, 28, 1]).astype('float32')
        else:
            # Normalize data
            X = X / 255
        y = pd.read_csv(labels_path_full)
        # Encode labels
        if one_hot_enc:
            encoder = OneHotEncoder(sparse=False, categories='auto')
            y = encoder.fit_transform(y)
        # Divide into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        if train_val_test:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3,
                                                              random_state=self.random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test
