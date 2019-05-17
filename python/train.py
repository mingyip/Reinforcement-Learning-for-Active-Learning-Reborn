
#!/usr/bin/env python3.6

""" Front-end script for training a RL agent. """

import tensorflow as tf
import numpy as np
import datetime
import random
import time
import json
import sys
import os


from learningai.agent.valueAgent import valueAgent
from learningai.env.mnist_env import mnist_env


def main():

    np.set_printoptions(precision=6)
    tf.random.set_random_seed(10)
    np.random.seed(10)

    sess = tf.Session()
    cnn_env = mnist_env(sess, lr=1e-4)
    dqn_agent = valueAgent(sess, cnn_env, lr=1e-3, gamma=0.9)


    sess.run(tf.global_variables_initializer())
    cnn_env.storeLocalVar()
    cnn_env.resetNetwork()
    print(cnn_env.get_validation_accuracy())

    print("End of Training")

if __name__ == '__main__':
    main()