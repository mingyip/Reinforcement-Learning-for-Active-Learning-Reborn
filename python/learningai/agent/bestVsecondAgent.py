import numpy as np


from learningai.agent.model.mnist64x5 import mnist64x5_model
from config import Config


class bestVsecondAgent(object):
    def __init__(self, sess, env, logger, num_class=10):

        print("Create a Best vs Second Best Agent!")
        self.num_class = num_class
        self.sess = sess
        self.env = env
        self.logger = logger

    def train(self):
        """ We pick the largest best-second best sets to train """

        # Set Constant/ Params
        budget          = Config.EVALUATION_CLASSIFICATION_BUDGET
        episodes        = Config.AGENT_TRAINING_EPISODES
        epochs          = Config.EVALUATION_CLASSIFICATION_EPOCH
        selection_size  = Config.EVALUATION_SELECTION_BATCHSIZE
        train_size      = Config.EVALUATION_TRAINING_BATCHSIZE
        validation_imgs = 1500
        test_imgs       = -1

        # Set training array and variable
        S               = np.zeros((selection_size, self.num_class+2))
        counter         = 0
        reward_sum      = 0

        for episode in range(episodes):
            self.begin_episode()
            counter = 0

            for iteration in range(int(budget/train_size)):
                ntrained        = iteration * train_size
                remain_budget   = (budget - ntrained) / budget
                remain_episodes = (episodes - episode) / episodes

                [x_select, y_select, idx] = self.env.get_next_selection_batch()
                S[:, 0:-2] = self.get_next_state_from_env(x_select)
                S[:, -2] = remain_budget
                S[:, -1] = remain_episodes

                train_idx = self.get_train_set(S[:, 0:-2])
                self.train_env(x_select[train_idx], y_select[train_idx], epochs)
                counter = counter + len(train_idx)

                reward = self.get_validation_accuracy(1000)
                print("Eps:", episode, " Iter:", iteration, " Reward:", reward, end="\r")
            reward = self.get_test_accuracy()
            reward_sum = reward_sum + reward
            print(str.format('Eps:{0:3.0f} R:{1:.4f} Size: {2:3.0f}                          ', episode, reward, counter))
            # print(str.format('dist:{0:3.0f} {1:3.0f} {2:3.0f} {3:3.0f} {4:3.0f} {5:3.0f} {6:3.0f} {7:3.0f} {8:3.0f} {9:3.0f}', dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], dist[6], dist[7], dist[8], dist[9]))

        print("Mean: ", reward_sum/episodes)

    def begin_episode(self):
        """ Reset the classificaiton network """
        self.env.resetNetwork()

    def reset_network(self):
        """ Reset the classification network """
        # TODO: maybe merge with the function begin_episode
        self.env.resetNetwork()

    def get_next_state_from_env(self, imgs):
        """ Get output probability of new images from the environment """
        state = self.env.get_output_probability(imgs)
        return state

    def get_train_set(self, select_set):
        """ Pick training set from the selection set """
        
        train_size = Config.TRAINING_BATCHSIZE

        sorted_set = np.sort(select_set)
        scores = sorted_set[:, -1] - sorted_set[:, -2]
        ranked = np.argsort(scores)
        # lows = ranked[-train_size:]
        tops = ranked[0:train_size]

        # print(sorted_set)
        # print(scores)
        # print(ranked)
        # print(tops)
        # raise

        return tops


        # ranked = np.argsort(predict)
        # top_idx = ranked[-Config.TRAINING_BATCHSIZE:]
        # low_idx = ranked[:-Config.TRAINING_BATCHSIZE]
        # raise 
        # return np.arange(train_size)

    def train_env(self, x_train, y_train, epoch):
        """ Train classification network with dataset """
        self.env.train_env(x_train, y_train, epoch)

    def get_validation_accuracy(self, nImages=-1):
        """ Get validation reward from the environment """
        return self.env.get_validation_accuracy(nImages)

    def get_test_accuracy(self, nImages=-1):
        """ Get test reward from the environment """
        return self.env.get_test_accuracy(nImages)

    def store_network_var(self):
        pass

    def reset_network(self):
        pass