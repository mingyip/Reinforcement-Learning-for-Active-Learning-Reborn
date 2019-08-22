import tensorflow as tf
import numpy as np
import datetime
import random
import time
import json
import sys
import os


from learningai.utils.AgentLogger import AgentLogger
from utils.loggingManger import loggingManger
from config import Config


def test():
    start = time.time()
    # tf.logging.set_verbosity(tf.logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.set_printoptions(precision=6)
    tf.random.set_random_seed(Config.TF_SEED)
    np.random.seed(Config.NP_SEED)

    logger = loggingManger()


    done = time.time()
    elapsed = done - start
    print(elapsed)
    logger.log([["Time elapsed", elapsed]])


    logs = []

    for i in range(10):

        log = {
            "episode":      None,
            "top_reward":   i,
            "exp_rate":     None,
            "trainsize":    i,
            "top_dist":     [1+i, 2, 3],
            "top_pred":     [3+i, 2, 1]
        }
        logs.append(log)

    AgentLogger.log_training_init(logger)
    AgentLogger.log_training_results(logs, logger)
    AgentLogger.log_evaluation_results(logs, logger, 10, 2, 0.5)



if __name__ == '__main__':
    test()