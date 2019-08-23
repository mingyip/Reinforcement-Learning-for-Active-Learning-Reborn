import tensorflow as tf
import numpy as np
import copy

from learningai.utils.AgentLogger import AgentLogger
from learningai.agent.model.mnist64x5 import mnist64x5_model
# from utils.loggingManger import loggingManger
from config import Config

class valueAgent(object):
    def __init__(self, sess, env, logger, lr=1e-3, gamma=0.9, num_class=10):

        print("Create a State Value Estimation agent!")
        self.num_class = num_class
        self.sess = sess
        self.env = env
        self.logger = logger
        # super(valueAgent, self).__init__(n_class=self.env.nclass, lr=lr, gamma=gamma, scopeName="dqn_mnist64x5")
        self.dqn_init       = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_init")
        self.dqn            = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_train")
        self.dqn_best       = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_best")
        self.memory         = []
        self.logs           = []
        self.best_reward    = -1
        self.best_episode   = -1

    def train(self):
        """ Reinforcement Learning Algorithm """
        # TODO: Fix the second remain_budget and remain_episodes bug.
        # DONE:  avg should move the gpu to calulate
        # TODO: What about the terminal state. I dont get the K+1 state. How do I store the S, S_new pair.
        # TODO: Re-implement experience replay
        # TODO: Re-implement target network
        # TODO: Re-implement early stop and late start
 
        # Set Constant
        budget                  = Config.CLASSIFICATION_BUDGET
        episodes                = Config.AGENT_TRAINING_EPISODES
        epochs                  = Config.CLASSIFICATION_EPOCH
        selection_size          = Config.SELECTION_BATCHSIZE
        train_size              = Config.TRAINING_BATCHSIZE
        train_start_at          = Config.TRAINING_START_AT_ITER
        train_end_at            = Config.TRAINING_FINISH_AT_ITER
        exporation_rate         = 1.0
        exporation_decay_rate   = Config.EXPLORATION_DECAY_RATE
        validation_images       = 1500
        is_experience_replay    = Config.USE_EXPERIENCE_REPLAY

        # Set array and variable
        S                       = np.zeros((selection_size, self.num_class+2))
        S_new                   = np.zeros((selection_size, self.num_class+2))
        S_old                   = np.zeros((selection_size, self.num_class+2))
        last_remain_budget      = None
        last_remain_episodes    = None
        last_status             = np.zeros((selection_size, self.num_class+2))
        train_steps             = 0

        AgentLogger.log_training_init(self.logger)
        for episode in range(episodes):
            self.evaluate_agent(episode, exporation_rate)
            self.begin_episode()

            for iteration in range(int(budget/train_size)):
                ntrained        = iteration * train_size
                remain_budget   = (budget - ntrained) / budget
                remain_new_bgt  = max(budget-ntrained-1, 0) / budget
                remain_episodes = (episodes - episode) / episodes

                # Get New States
                [x_select, y_select, select_idx] = self.env.get_next_selection_batch()
                S[:, 0:-2] = self.get_next_state_from_env(x_select)
                S[:, -2] = remain_budget
                S[:, -1] = remain_episodes

                # Exporation vs Exploitation
                # if (np.random.rand(1)[0]>exporation_rate):
                #     predicts, _, train_idx, _ = self.predict(S)
                # else:
                train_idx = np.arange(train_size)

                # Train Classification Network
                batch_x = x_select[train_idx]
                batch_y = y_select[train_idx]
                self.train_env(batch_x, batch_y, epochs)

                # Train DQN Network
                [x_new_select, y_new_select, select_idx] = self.env.get_next_selection_batch(peek=True)
                S_new[:, 0:-2] = self.get_next_state_from_env(x_new_select)
                S_new[:, -2] = remain_new_bgt
                S_new[:, -1] = remain_episodes

                predicts_new, tops_new, _, _ = self.predict(S_new)
                avg_V = np.mean(predicts_new[tops_new])
                reward = self.get_environment_accuracy(nImages=validation_images)

                self.train_agent(reward, S[train_idx], avg_V)
                train_steps = train_steps + 1
                print("Eps:", episode, " Iter:", iteration, " Reward:", reward, end="\r")

            self.write_all_logs()
            self.reinitialize_log_batch()
            if exporation_rate > 0:
                exporation_rate -= exporation_decay_rate

        self.evaluate_agent(episodes, exporation_rate)
        self.evaluate_best_agent()

    def begin_game(self):
        """ Reset the agent in the game """
        self.reset_network()
        self.env.resetEnvCounter()

    def begin_episode(self):
        """ Reset the agent memory and the environment """
        # DONE: reset the agent memory
        self.env.resetNetwork()
        self.memory = []

    def reset_network(self):
        """ Reset the classification network """
        # TODO: maybe merge with the function begin_episode
        self.env.resetNetwork()

    def get_next_state_from_env(self, imgs):
        """ Get output probability of new images from the environment """
        y = self.env.get_output_probability(imgs)
        return y

    def get_environment_accuracy(self, nImages=-1, isValidation=True):
        """ Get reward from the environment """
        if isValidation:
            reward = self.env.get_validation_accuracy(nImages)
        else:
            reward = self.env.get_test_accuracy(nImages)
        return reward

    def train_agent(self, reward, state, V_new):
        """ Train agent with the TD-error """
        R  = np.full((Config.TRAINING_BATCHSIZE, 1), reward)
        V_ = np.full((Config.TRAINING_BATCHSIZE, 1), V_new)
        feed_dict = {self.dqn.R:R, self.dqn.s:state, self.dqn.avg_V_:V_}

        self.sess.run(self.dqn.train_op, feed_dict)

    def train_env(self, x_train, y_train, epoch):
        """ Train classification network with dataset """
        self.env.train_env(x_train, y_train, epoch)

    def predict(self, state, batchsize=None):
        """ Agent predicts the state-action value (accuracy) """
        # TODO: function predict write log directly to the result.log
        # TODO: Move the calculation steps to GPU

        if batchsize is None:
            batchsize = Config.TRAINING_BATCHSIZE
    
        feed_dict = {self.dqn.s: state}
        predict = (self.sess.run(self.dqn.V, feed_dict)).squeeze()

        ranked = np.argsort(predict)
        top_idx = ranked[-batchsize:]
        low_idx = ranked[0:batchsize]

        return predict, top_idx, low_idx, ranked

    def reinitialize_log_batch(self):
        self.logs = []

    def print_all_logs(self):
        AgentLogger.print_trianing_results(self.logs)

    def write_all_logs(self):
        AgentLogger.log_training_results(self.logs, self.logger)

    def evaluate_agent(self, episode, exp_rate, isValidation=True, isStream=True):

        low_reward = low_dist = low_size = low_pred = None
        [top_reward, top_dist, top_size, top_pred] = self.evaluate(isStream=True)
        # if evalLow:
        #     [low_reward, low_dist, low_size, low_pred] = self.evaluate(isStream=True, trainTop=False)

        log = {
            "episode":      episode,
            "top_reward":   top_reward,
            "exp_rate":     exp_rate,
            "trainsize":    top_size,
            "top_dist":     top_dist,
            "top_pred":     top_pred,
            "low_reward":   low_reward,
            "low_dist":     low_dist,
            "low_pred":     low_pred
        }

        self.logs.append(log)

        if top_reward > self.best_reward:
            self.best_reward = top_reward
            self.best_episode = episode
        self.store_best_network_var()

        AgentLogger.print_trianing_results(log)

    def evaluate(self, isValidation=True, isStream=True):
        """ Agent uses the current policy to train a new network """
        """ Both stream and pool based learning """
        """ Here we create re-initialize the env model and train it """
        budget              = Config.EVALUATION_CLASSIFICATION_BUDGET
        epochs              = Config.EVALUATION_CLASSIFICATION_EPOCH
        selection_size      = Config.EVALUATION_SELECTION_BATCHSIZE
        train_size          = Config.EVALUATION_TRAINING_BATCHSIZE
        start_rank          = Config.EVALUATION_START_RANK
        iterations          = int(budget/train_size)
        S                   = np.zeros((selection_size, self.num_class+2))
        remain_episodes     = 0
        num_imgs            = -1
        distribution        = np.zeros((self.num_class))
        total_train_size    = 0

        self.reset_network()
        for iteration in range(iterations):

            ntrained        = iteration * train_size
            remain_budget   = (budget - ntrained) / budget

            [x_select, y_select, idx] = self.env.get_next_selection_batch(batchsize=selection_size)
            S[:, 0:-2] = self.get_next_state_from_env(x_select)
            S[:, -2] = remain_budget
            S[:, -1] = remain_episodes
            predicts, tops, lows, ranked = self.predict(S, batchsize=train_size)

            # temp = np.argmax(y_select, axis=1)
            # unique, counts = np.unique(temp, return_counts=True)

            train_idx = ranked[start_rank : start_rank+train_size]
            batch_x = x_select[train_idx]
            batch_y = y_select[train_idx]
            total_train_size = total_train_size + len(train_idx)
            distribution = distribution + np.sum(batch_y, axis=0)
            self.train_env(batch_x, batch_y, epochs)

        reward = self.get_environment_accuracy(num_imgs, isValidation)
        return [reward, distribution, total_train_size, predicts[train_idx]]

    def evaluate_best_agent(self):
        """ Evaluate the best agent """

        eval_eps   = Config.EVALUATION_EPISODES
        reward_sum = 0
        log_list = []

        print("Evaluate The Best Network with Test data: episode ", self.best_episode, " reward ", self.best_reward)
        self.restore_best_network_to_train_network()

        for i in range(eval_eps):
            [reward, dist, trainsize, pred] = self.evaluate(isValidation=False)
            reward_sum = reward_sum + reward
            log = {
                "episode":      None,
                "top_reward":   reward,
                "exp_rate":     None,
                "trainsize":    trainsize,
                "top_dist":     dist,
                "top_pred":     pred
            }
            log_list.append(log)

            AgentLogger.print_trianing_results(log)

        mean_reward = reward_sum/eval_eps
        print("Mean: ", mean_reward)
        AgentLogger.log_evaluation_results(log_list, self.logger, self.best_episode, self.best_reward, mean_reward)

    def store_network_var(self):
        """ Store network variables so that later we can re-init the network """

        tf_vars = tf.trainable_variables()
        self.dqn_init_var   = [var for var in tf_vars if 'dqn_init' in var.name]
        self.dqn_train_var  = [var for var in tf_vars if 'dqn_train' in var.name]
        self.dqn_best_var   = [var for var in tf_vars if 'dqn_best' in var.name]
        self.reset_optimizer_op = tf.variables_initializer(self.dqn.optimizer.variables())

    def store_best_network_var(self):
        """ save the best network variables """
        # TODO: Check if the optimizer get copied.
        for idx, var in enumerate(self.dqn_train_var):
            self.sess.run(tf.assign(self.dqn_best_var[idx], var))

    def restore_best_network_to_train_network(self):
        """ restore the best network to train network """
        # TODO: Check if the optimizer get copied.
        for idx, var in enumerate(self.dqn_best_var):
            self.sess.run(tf.assign(self.dqn_train_var[idx], var))


