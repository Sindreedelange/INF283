# Works as long as Keras is in the activated environment
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import regularizers
import keras

from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import time


class Models:

    def __init__(self):
        self.model_path = "../models/"
        self.optimizer = Adam

    # CNN model
    # -----------------------------------------------------------------------
    def get_cnn_model(self, learning_rate=1e-3, loss='categorical_crossentropy',
                      metrics='accuracy', dropout=0.4, pool_size=(3,3),
                      activation='relu',
                      ):
        self.optimizer = Adam(lr=learning_rate)
        model_cnn = Sequential()
        model_cnn.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation=activation))
        model_cnn.add(MaxPooling2D(pool_size=(pool_size)))
        model_cnn.add(Dropout(dropout))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(128, activation=activation))
        model_cnn.add(Dense(10, activation='softmax'))
        model_cnn.compile(loss=loss, optimizer=self.optimizer, metrics=[metrics])
        return model_cnn

    def train_cnn_model(self, model,
                                X_train,
                                X_val,
                                X_test,
                                y_train,
                                y_val,
                                y_test,
                                epochs=4,
                                bs=64):
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=bs)
        end_time = time.time()
        tot_time = round((end_time - start_time), 3)
        return tot_time, history

    # -----------------------------------------------------------------------

    # NN model
    # -----------------------------------------------------------------------
    def train_nn_model(self, model,
                                X_train,
                                X_val,
                                X_test,
                                y_train,
                                y_val,
                                y_test,
                                epochs=4,
                                bs=64):
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=bs)
        end_time = time.time()
        tot_time = round((end_time - start_time), 3)
        return tot_time, history

    def get_nn_model(self, learning_rate=1e-3, regularizer=0.01,
                     loss='categorical_crossentropy', metrics='accuracy'):
        optimizer = self.optimizer(lr=learning_rate)
        regularizer_l2 = regularizers.l2(regularizer)
        model_nn = Sequential()
        model_nn.add(Dense(256, input_dim=784, activation='relu', kernel_regularizer=regularizer_l2))
        model_nn.add(Dense(128, activation='relu', kernel_regularizer=regularizer_l2))
        model_nn.add(Dense(60, activation='relu', kernel_regularizer=regularizer_l2))
        model_nn.add(Dense(10, activation='softmax'))
        model_nn.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        return model_nn
    # -----------------------------------------------------------------------

    def save_model(self, model, model_name, extension='.hdf5'):
        print("Saving model: ", model_name, " to the folder: ", self.model_path)
        full_model_path = self.model_path + model_name + extension
        keras.models.save_model(model, full_model_path)

    def load_model(self, model_name, extension='.hdf5'):
        full_path = self.model_path + model_name + extension
        return keras.models.load_model(full_path)
