
import tensorflow as tf
import numpy as np

from learningai.env.model.mnist_cnn import mnist_cnn
from config import Config

class mnist_env(object):

    def __init__(self, sess, lr=1e-4):

        print("Create a Mnist Classification Environment!")
        self.loadMnist()

        self.sess = sess
        self.cnn_init = mnist_cnn(self.width, self.height, self.nclass, lr=lr, scopeName="cnn_init")
        self.cnn = mnist_cnn(self.width, self.height, self.nclass, lr=lr, scopeName="cnn_train")


    def loadMnist(self):
        """
        Load Mnist dataset and do some data pre-processing
        Split the training set 80/20% for training and validation set
        Convert y labels to 1-k hot array
        """
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.ncolor = 1 if len(x_train.shape)==3 else x_train.shape[3]
        self.nclass = np.max(y_test) + 1
        self.width  = x_train.shape[1]
        self.height = x_train.shape[2]
        self.total_train_size = len(x_train)
        self.ntrain = int(0.9 * self.total_train_size)
        self.nval = int(0.1 * self.total_train_size)
        self.ntest  = len(x_test)
        self.train_counter = 0
        self.train_index = np.arange(self.ntrain)

        x_train = x_train/255.0
        x_train = np.reshape(x_train, (self.total_train_size, -1))
        x_test = x_test/255.0
        x_test = np.reshape(x_test, (self.ntest, -1))

        labels = np.zeros((self.total_train_size, self.nclass))
        labels[np.arange(self.total_train_size), y_train] = 1
        y_train = labels

        x_val = x_train[self.ntrain:self.total_train_size]
        y_val = y_train[self.ntrain:self.total_train_size]

        x_train = x_train[0:self.ntrain]
        y_train = y_train[0:self.ntrain]

        labels = np.zeros((self.ntest, self.nclass))
        labels[np.arange(self.ntest), y_test] = 1
        y_test = labels

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


    def train_env(self, x_train, y_train, epoch):
        feed_dict = {self.cnn.x:x_train, self.cnn.y_:y_train}
        for i in range(epoch):
            self.sess.run(self.cnn.train_op, feed_dict)

    def train_env_with_idx(self, idx, epoch):
        self.train_env(self.x_train[idx], self.y_train[idx], epoch)

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
        [y, fc1] = self.sess.run([self.cnn.y, self.cnn.fc1], {self.cnn.x: x_train})
        return y

    def get_validation_accuracy(self, nImages=-1):
        feed_dict = {self.cnn.x:self.x_val[0:nImages], self.cnn.y_:self.y_val[0:nImages]}
        acc = self.sess.run(self.cnn.accuracy, feed_dict)
        return acc

    def get_test_accuracy(self, nImages=-1):
        feed_dict = {self.cnn.x:self.x_test[0:nImages], self.cnn.y_:self.y_test[0:nImages]}
        acc = self.sess.run(self.cnn.accuracy, feed_dict)
        return acc

    def storeNetworkVar(self):
        """ Store network variables so that later we can re-init the network """

        tf_vars = tf.trainable_variables()
        self.cnn_init_var = [var for var in tf_vars if 'cnn_init' in var.name]
        self.cnn_train_var = [var for var in tf_vars if 'cnn_train' in var.name]
        self.reset_optimizer_op = tf.variables_initializer(self.cnn.optimizer.variables())

    def resetNetwork(self):
        """ Re-init cnn_var with values of cnn_init_var """

        for idx, var in enumerate(self.cnn_init_var):
            self.sess.run(tf.assign(self.cnn_train_var[idx], var))
        self.sess.run(self.reset_optimizer_op)

    def resetEnvCounter(self):
        """ Environment image counter """
        self.train_counter = 0
        self.train_index = np.arange(self.ntrain)