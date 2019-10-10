
import tensorflow as tf
import numpy as np
import keras
import os


from emnist import list_datasets
from emnist import extract_training_samples, extract_test_samples
from learningai.env.model.emnist_cnn import emnist_cnn
from config import Config

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

class emnist_env(object):

    def __init__(self, sess, lr=1e-4):
        print("Create a Emnist Classification Environment!")
        self.loadEmnist()

        self.sess = sess
        self.cnn_init = emnist_cnn(self.x_train.shape[1:], self.nclass)
        self.cnn      = emnist_cnn(self.x_train.shape[1:], self.nclass)

    def loadEmnist(self):
        """
        Load Emnist dataset and do some data pre-processing
        Split the training set 80/20% for training and validation set
        Convert y labels to 1-k hot array
        """

        x_train, y_train = extract_training_samples('balanced')
        x_test, y_test   = extract_test_samples('balanced')

        # Get only the upper case letters
        train_alphabet_list = (np.array(y_train) < 36) & (np.array(y_train) > 9)
        test_alphabet_list  = (np.array(y_test) < 36) & (np.array(y_test) > 9)

        y_train = y_train[train_alphabet_list] - 10
        x_train = x_train[train_alphabet_list]
        y_test = y_test[test_alphabet_list] - 10
        x_test = x_test[test_alphabet_list]

        self.nclass = 26
        self.width  = x_train.shape[1]
        self.height = x_train.shape[2]
        self.total_train_size = len(x_train)
        self.ntrain = int(0.9 * self.total_train_size)
        self.nval = int(0.1 * self.total_train_size)
        self.ntest  = len(x_test)
        self.train_counter = 0
        self.train_index = np.arange(self.ntrain)

        x_train = x_train.reshape(x_train.shape[0], self.width, self.height, 1)
        x_test = x_test.reshape(x_test.shape[0], self.width, self.height, 1)
        input_shape = (self.width, self.height, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        self.x_test = x_test/255

        self.x_val = x_train[self.ntrain:self.total_train_size]
        self.x_train = x_train[0:self.ntrain]
        y_val = y_train[self.ntrain:self.total_train_size]
        y_train = y_train[0:self.ntrain]

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(y_train, 26)
        self.y_val = keras.utils.to_categorical(y_val, 26)
        self.y_test = keras.utils.to_categorical(y_test, 26)

        print(self.x_train.shape)
        print(self.x_val.shape)
        print(self.x_test.shape)

    def train_env(self, x_train, y_train, epochs):
        self.cnn.model.fit(x_train, y_train, 
                        batch_size  = len(x_train),
                        epochs      = epochs,
                        verbose     = 0)

    def train_env_with_idx(self, idx, epochs):
        self.cnn.model.fit(self.x_train, self.y_train, 
                        batch_size  = 50,
                        epochs      = epochs,
                        verbose     = 0)


    def get_next_selection_batch(self, batchsize=None, peek=False):
        """ Return the mext image batchfor selection step """
        # DONE:  After the selection is full, reshuffle the batch
        # DONE:  Use another array to store the original img batch to retain the order?
        # TODO: np.random.shuffle makes peek not static
        # TODO: Refactor the code after we implement experience replay and multi-step

        if batchsize is None:
            batchsize = Config.SELECTION_BATCHSIZE 

        if (self.train_counter + batchsize > self.ntrain):
            bgn_idx = 0
            end_idx = batchsize
            if not peek:
                self.train_counter = 0
            np.random.shuffle(self.train_index)
        else:
            bgn_idx = self.train_counter
            end_idx = self.train_counter + batchsize
            if not peek:
                self.train_counter = end_idx

        idx     = self.train_index[bgn_idx:end_idx]
        x_batch = self.x_train[idx]
        y_batch = self.y_train[idx]

        # print("idx:", self.train_counter, self.train_index[bgn_idx:end_idx])
        return [x_batch, y_batch, idx]

    def get_next_train_batch(self, selection_batchsize=None, train_batchsize=None):
    
        if train_batchsize is None:
            train_batchsize = Config.TRAINING_BATCHSIZE
        
        if selection_batchsize is None:
            selection_batchsize = Config.SELECTION_BATCHSIZE

        if (self.train_counter + selection_batchsize > self.ntrain):
            bgn_idx = 0
            end_idx = selection_batchsize
            self.train_counter = 0
            np.random.shuffle(self.train_index)
        else:
            bgn_idx = self.train_counter
            end_idx = self.train_counter + selection_batchsize
            self.train_counter = end_idx

        randList = np.arange(selection_batchsize)
        np.random.shuffle(randList)
        idx = (self.train_index[bgn_idx:end_idx])[randList[:train_batchsize]]

        x_batch = self.x_train[idx]
        y_batch = self.y_train[idx]  
        return [x_batch, y_batch, idx]   

    def get_output_probability(self, x_train):
        """ 
        Feed x_train into the network and 
        return the output probability of the final layer 
        """
        return self.cnn.model.predict(x_train)

    def get_validation_accuracy(self, nImages=-1):
        return self.cnn.model.evaluate(self.x_val[0:nImages], self.y_val[0:nImages], verbose=0)[1]

    def get_test_accuracy(self, nImages=-1):
        return self.cnn.model.evaluate(self.x_test[0:nImages], self.y_test[0:nImages], verbose=0)[1]

    def storeNetworkVar(self):
        """ Store network variables so that later we can re-init the network """
        #TODO: reset the whole rmsp
        self.cnn_init_var   = self.cnn_init.model.get_weights()

    def resetNetwork(self):
        """ Re-init cnn_var with values of cnn_init_var """
        self.cnn.model.set_weights(self.cnn_init_var)
        self.cnn.reset_op()

    def resetEnvCounter(self):
        """ Environment image counter """
        self.train_counter = 0
        self.train_index = np.arange(self.ntrain)
