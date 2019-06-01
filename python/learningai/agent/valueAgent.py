import tensorflow as tf
import numpy as np


from learningai.agent.model.mnist64x5 import mnist64x5_model
from config import Config

class valueAgent(object):
    def __init__(self, sess, env, lr=1e-3, gamma=0.9):

        print("Create a State Value Estimation agent!")
        self.sess = sess
        self.env = env
        # super(valueAgent, self).__init__(n_class=self.env.nclass, lr=lr, gamma=gamma, scopeName="dqn_mnist64x5")
        self.dqn_init = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_init")
        self.dqn = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_train")

    def train(self):
        """ Reinforcement Learning Algorithm """
        # TODO: Fix the second remain_budget and remain_episodes bug.
        # TODO: avg should move the gpu to calulate

        budget                  = Config.CLASSIFICATION_BUDGET
        episodes                = Config.AGENT_TRAINING_EPISODES
        epochs                  = Config.CLASSIFICATION_EPOCH
        selection_size          = Config.SELECTION_BATCHSIZE
        train_size              = Config.TRAINING_BATCHSIZE
        exporation_rate         = 1.0
        exporation_decay_rate   = 0.005

        for episode in range(episodes):
            self.begin_episode()
            reward = self.get_last_env_reward(nImages=1000)

            for iteration in range(int(budget/train_size)):
                ntrained        = iteration * train_size
                remain_budget   = (budget - ntrained) / budget
                remain_episodes = (episodes - episode) / episodes                

                if (np.random.rand(1)[0]>exporation_rate):
                    # Predict Rewards of different actions
                    [x_select, y_select] = self.env.get_next_selection_batch()
                    S = self.get_next_state_from_env(x_select)

                    # Select the best k-th images
                    predicts, tops, _ = self.predict(S, remain_budget, remain_episodes)
                    batch_x = x_select[tops]
                    batch_y = y_select[tops]
                else:
                    [batch_x, batch_y] = self.env.get_next_train_batch()

                # Train Classification Network
                self.train_env(batch_x, batch_y, epochs)

                # Train DQN Network
                [x_new_select, y_new_select] = self.env.get_next_selection_batch(peek=True)
                S_new = self.get_next_state_from_env(x_new_select)
                predicts_new, tops_new, _ = self.predict(S_new, remain_budget, remain_episodes)
                avg_V = np.mean(predicts_new[tops_new])
                reward = self.get_last_env_reward(nImages=1000)

                self.train_agent(reward, S_new[tops_new], remain_budget, remain_episodes, avg_V)
                print("episode:", episode, "  iteration:", iteration, "reward: ", reward, "exporation_rate:", exporation_rate)

            if exporation_rate > 0:
                exporation_rate -= exporation_decay_rate

    def begin_game(self):
        """ Reset the agent in the game """
        self.resetNetwork()
        self.env.resetEnvCounter()

    def begin_episode(self):
        """ Reset the agent memory and the environment """
        # TODO: reset the agent memory
        self.env.resetNetwork()

    def get_next_state_from_env(self, imgs):
        """ Get output probability of new images from the environment """
        state = self.env.get_output_probability(imgs)
        return state

    def get_last_env_reward(self, nImages=-1):
        """ Get last validation reward from the environment """
        return self.env.get_validation_accuracy(nImages)

    def train_agent(self, reward, state, remainBudget, remainEpisode, V_new):
        """ Train agent with the TD-error """
        R  = np.full((Config.TRAINING_BATCHSIZE, 1), reward)
        B  = np.full((Config.TRAINING_BATCHSIZE, 1), remainBudget)
        E  = np.full((Config.TRAINING_BATCHSIZE, 1), remainEpisode)
        V_ = np.full((Config.TRAINING_BATCHSIZE, 1), V_new)
        feed_dict = {self.dqn.R:R, self.dqn.b:B, self.dqn.e:E, self.dqn.s:state, self.dqn.avg_V_:V_}

        self.sess.run(self.dqn.train_op, feed_dict)

    def train_env(self, x_train, y_train, epoch):
        """ Train classification network with dataset """
        self.env.train_env(x_train, y_train, epoch)

    def predict(self, state, remainBudget, remainEpisode):
        """ Agent predicts the state-action value (accuracy) """
        # TODO: function predict write log directly to the result.log
        # TODO: Move the calculation steps to GPU

        budget = np.full((Config.SELECTION_BATCHSIZE, 1), remainBudget, dtype=float)
        episode = np.full((Config.SELECTION_BATCHSIZE, 1), remainEpisode, dtype=float)
        feed_dict = {self.dqn.s: state, self.dqn.b: budget, self.dqn.e: episode}
        predict = (self.sess.run(self.dqn.V, feed_dict)).squeeze()

        ranked = np.argsort(predict)
        top_idx = ranked[-Config.TRAINING_BATCHSIZE:]
        low_idx = ranked[:-Config.TRAINING_BATCHSIZE]

        return predict, top_idx, low_idx

    def evaluate(self, isStream):
        """ Agent uses the current policy to train a new network """
        """ Both stream and pool based learning """
        pass

    def storeNetworkVar(self):
        """ Store network variables so that later we can re-init the network """

        tf_vars = tf.trainable_variables()
        self.dqn_init_var = [var for var in tf_vars if 'dqn_init' in var.name]
        self.dqn_train_var = [var for var in tf_vars if 'dqn_train' in var.name]
        self.reset_optimizer_op = tf.variables_initializer(self.dqn.optimizer.variables())

    def resetNetwork(self):
        """ Re-init cnn_var with values of cnn_init_var """

        for idx, var in enumerate(self.dqn_init_var):
            self.sess.run(tf.assign(self.dqn_train_var[idx], var))
        self.sess.run(self.reset_optimizer_op)
