
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


from learningai.agent.bestVsecondAgent import bestVsecondAgent as bvsAgent
from learningai.agent.valueAgent import valueAgent
from learningai.env.mnist_env import mnist_env
from utils.loggingManger import loggingManger
from config import Config


def main():

    start = time.time()
    # tf.logging.set_verbosity(tf.logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.set_printoptions(precision=6)
    tf.random.set_random_seed(Config.TF_SEED)
    np.random.seed(Config.NP_SEED)

    logger = loggingManger()
    sess = tf.Session()
    cnn_env = mnist_env(sess, lr=1e-4)
    
    agent_type = Config.AGENT_TYPE
    if agent_type == "valueAgent":
        agent = valueAgent(sess, cnn_env, logger, lr=1e-3, gamma=0.9)
    elif agent_type == "BVSB":
        agent = bvsAgent(sess, cnn_env, logger)

    sess.run(tf.global_variables_initializer())
    cnn_env.storeNetworkVar()
    cnn_env.resetNetwork()
    agent.store_network_var()
    agent.reset_network()

    agent.train()
    print("End of Training")

    done = time.time()
    elapsed = done - start
    print(elapsed)
    logger.log(["Time elapsed", elapsed])

if __name__ == '__main__':
    main()