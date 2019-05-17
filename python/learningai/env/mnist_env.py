
import tensorflow as tf
import numpy as np

from learningai.env.model.mnist_cnn import mnist_cnn

class mnist_env(object):

    def __init__(self, sess, lr=1e-4):

        print("Create a Mnist Classification Environment!")
        self.loadMnist()

        self.sess = sess
        self.cnn_init = mnist_cnn(self.width, self.height, self.nclass, lr=lr, scopeName="cnn_init")
        self.cnn = mnist_cnn(self.width, self.height, self.nclass, lr=lr, scopeName="cnn_train")


    def loadMnist(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.ncolor = 1 if len(x_train.shape)==3 else x_train.shape[3]
        self.nclass = np.max(y_test) + 1
        self.width  = x_train.shape[1]
        self.height = x_train.shape[2]
        self.total_train_size = len(x_train)
        self.ntrain = int(0.8 * self.total_train_size)
        self.nval = int(0.2 * self.total_train_size)
        self.ntest  = len(x_test)

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


    def get_validation_accuracy(self):
        acc = self.sess.run(self.cnn.accuracy, {self.cnn.x:self.x_val, self.cnn.y_:self.y_val})
        return acc


    def storeLocalVar(self):
        tf_vars = tf.trainable_variables()
        self.cnn_init_var = [var for var in tf_vars if 'cnn_init' in var.name]
        self.cnn_var = [var for var in tf_vars if 'cnn_train' in var.name]
        self.reset_optimizer_op = tf.variables_initializer(self.cnn.optimizer.variables())

    def resetNetwork(self):
        # Re-init cnn_var with values of cnn_init_var
        for idx, var in enumerate(self.cnn_init_var):
            self.sess.run(tf.assign(self.cnn_var[idx], var))
        self.sess.run(self.reset_optimizer_op)