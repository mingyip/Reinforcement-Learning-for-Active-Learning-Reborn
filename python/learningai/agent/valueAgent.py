import tensorflow as tf
import numpy as np
import copy


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
        self.dqn_init   = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_init")
        self.dqn        = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_train")
        self.dqn_best   = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_best")
        self.memory     = []

    def train(self):
        """ Reinforcement Learning Algorithm """
        # TODO: Fix the second remain_budget and remain_episodes bug.
        # DONE:  avg should move the gpu to calulate
        # TODO: What about the terminal state. I dont get the K+1 state. How do I store the S, S_new pair.
 
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

        # Set array and variable
        S                       = np.zeros((selection_size, self.num_class+2))
        S_new                   = np.zeros((selection_size, self.num_class+2))
        S_old                   = np.zeros((selection_size, self.num_class+2))
        last_remain_budget      = None
        last_remain_episodes    = None
        last_status             = np.zeros((selection_size, self.num_class+2))
        best_reward             = -1
        best_episode            = -1


        # self.logger.log(["Episode", "Accuracy", "Train size", "Exporation Rate", "Dist", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None], newline=True)
        self.logger.log(["Episode", "Accuracy", "Train size", "Exporation Rate", "Dist", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "", "low Accuracy", "Dist", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], newline=True)
        for episode in range(episodes):
            self.begin_episode()
            [top_reward, top_dist, top_size] = self.evaluate(isStream=True, trainTop=True)
            [low_reward, low_dist, low_size] = self.evaluate(isStream=True, trainTop=False)
            self.log_training_results(episode, top_reward, exporation_rate, top_size, top_dist, low_reward, low_dist)

            for iteration in range(int(budget/train_size)):
                
                # print(int(budget/train_size))
                # print(train_end_at)
                # raise 
                # Stop training if >= train end
                # if iteration >= train_end_at:
                #     break

                ntrained        = iteration * train_size
                remain_budget   = (budget - ntrained) / budget
                remain_new_bgt  = max(budget-ntrained-1, 0) / budget
                remain_episodes = (episodes - episode) / episodes
                S_old           = copy.deepcopy(S)

                if (np.random.rand(1)[0]>exporation_rate):
                    # Predict Rewards of different actions
                    [x_select, y_select, select_idx] = self.env.get_next_selection_batch()

                    S[:, 0:-2] = self.get_next_state_from_env(x_select)
                    S[:, -2] = remain_budget
                    S[:, -1] = remain_episodes

                    # Select the best k-th images
                    predicts, train_idx, _ = self.predict(S)
                    batch_x = x_select[train_idx]
                    batch_y = y_select[train_idx]

                else:
                    [x_select, y_select, select_idx] = self.env.get_next_selection_batch()
                    S[:, 0:-2] = self.get_next_state_from_env(x_select)
                    S[:, -2] = remain_budget
                    S[:, -1] = remain_episodes

                    train_idx = np.arange(train_size)
                    batch_x = x_select[train_idx]
                    batch_y = y_select[train_idx]

                if iteration > 0:
                    pair = {"S":S_old, "select_idx":select_idx, "train_idx":train_idx, "S_":S, "R":reward}
                    self.memory.append(pair)

                # Train Classification Network
                self.train_env(batch_x, batch_y, epochs)

                # Train DQN Network
                [x_new_select, y_new_select, select_idx] = self.env.get_next_selection_batch(peek=True)
                S_new[:, 0:-2] = self.get_next_state_from_env(x_new_select)
                S_new[:, -2] = remain_new_bgt
                S_new[:, -1] = remain_episodes

                predicts_new, tops_new, _ = self.predict(S_new)
                avg_V = np.mean(predicts_new[tops_new])
                reward = self.get_validation_accuracy(nImages=validation_images)

                # Skip training if < train start
                # if iteration <  train_start_at: 
                #     continue
                self.train_agent(reward, S[train_idx], avg_V)
                print("Eps:", episode, " Iter:", iteration, " Reward:", reward, end="\r")

            [top_reward, top_dist, top_size] = self.evaluate(isStream=True, trainTop=True)
            [low_reward, low_dist, low_size] = self.evaluate(isStream=True, trainTop=False)
            self.log_training_results(episode, top_reward, exporation_rate, top_size, top_dist, low_reward, low_dist)

            if top_reward > best_reward:
                best_reward = top_reward
                best_episode = episode
                self.store_best_network_var()

            if exporation_rate > 0:
                exporation_rate -= exporation_decay_rate

        self.evaluate_best_agent(best_episode, best_reward)


    def begin_game(self):
        """ Reset the agent in the game """
        self.reset_network()
        self.env.resetEnvCounter()

    def begin_episode(self):
        """ Reset the agent memory and the environment """
        # DONE: reset the agent memory
        # print("value agent begin_episode")
        self.env.resetNetwork()
        self.memory = []

    def reset_network(self):
        """ Reset the classification network """
        # TODO: maybe merge with the function begin_episode
        self.env.resetNetwork()

    def get_next_state_from_env(self, imgs):
        """ Get output probability of new images from the environment """
        y = self.env.get_output_probability(imgs)

        # state = np.zeros((Config.SELECTION_BATCHSIZE, self.num_class+1024))
        # state[:, 0:10] = y
        # state[:, 10:10+1024] = fc
        return y

    def get_validation_accuracy(self, nImages=-1):
        """ Get validation reward from the environment """
        return self.env.get_validation_accuracy(nImages)

    def get_test_accuracy(self, nImages=-1):
        """ Get test reward from the environment """
        return self.env.get_test_accuracy(nImages)

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

        # budget = np.full((Config.SELECTION_BATCHSIZE, 1), remainBudget, dtype=float)
        # episode = np.full((Config.SELECTION_BATCHSIZE, 1), remainEpisode, dtype=float)
        feed_dict = {self.dqn.s: state}
        predict = (self.sess.run(self.dqn.V, feed_dict)).squeeze()

        ranked = np.argsort(predict)
        low_idx = ranked[-batchsize:]
        top_idx = ranked[0:batchsize]

        return predict, top_idx, low_idx

    def evaluate(self, isValidation=True, isStream=True, trainTop=True):
        """ Agent uses the current policy to train a new network """
        """ Both stream and pool based learning """
        budget          = Config.EVALUATION_CLASSIFICATION_BUDGET
        epochs          = Config.EVALUATION_CLASSIFICATION_EPOCH
        selection_size  = Config.EVALUATION_SELECTION_BATCHSIZE
        train_size      = Config.EVALUATION_TRAINING_BATCHSIZE
        iterations      = int(budget/train_size)
        S               = np.zeros((selection_size, self.num_class+2))
        remain_episodes = 0
        num_imgs        = -1
        distribution    = np.zeros((self.num_class))
        trainSize       = 0

        self.reset_network()
        # reward = self.get_validation_accuracy(num_imgs)
        # print("Reward: ", reward, "                          ", end='\r')
        for iteration in range(iterations):

            ntrained        = iteration * train_size
            remain_budget   = (budget - ntrained) / budget

            [x_select, y_select, idx] = self.env.get_next_selection_batch(batchsize=selection_size)
            S[:, 0:-2] = self.get_next_state_from_env(x_select)
            S[:, -2] = remain_budget
            S[:, -1] = remain_episodes
            predicts, tops, lows = self.predict(S, batchsize=train_size)

            if trainTop:
                train_idx = tops
            else:
                train_idx = lows

            batch_x = x_select[train_idx]
            batch_y = y_select[train_idx]
            trainSize = trainSize + len(train_idx)
            distribution = distribution + np.sum(batch_y, axis=0)

            self.train_env(batch_x, batch_y, epochs)

        if isValidation:
            reward = self.get_validation_accuracy(num_imgs)
        else:
            reward = self.get_test_accuracy(num_imgs)

        return [reward, distribution, trainSize]

    def evaluate_best_agent(self, best_episode, best_reward):
        """ Evaluate the best agent """

        eval_eps   = Config.EVALUATION_EPISODES
        reward_sum = 0

        print("Evaluate The Best Network with Test data: episode ", best_episode, " reward ", best_reward)
        self.logger.log(["Evaluate The Best Network with Test data:", "Episode", best_episode, "Accuracy", best_reward], newline=True)
        self.restore_best_network_to_train_network()

        for i in range(eval_eps):
            [reward, dist, trainsize] = self.evaluate(isValidation=False)
            reward_sum = reward_sum + reward
            self.log_training_results(None, reward, None, trainsize, dist)

        mean_reward = reward_sum/eval_eps
        print("Mean: ", mean_reward)
        self.logger.log(["Mean", mean_reward])

    def log_training_results(self, episode, reward, exp_rate, trainsize, distribution, low_reward=None, low_distribution=None):
        """ Write training logs to file """

        # Console Log
        epi_msg  = '        ' if episode is None else str.format('Eps:{0:3.0f} ', episode)
        rewd_msg = '         ' if reward is None else str.format('R:{0:.4f} ', reward)
        size_msg = '       ' if trainsize is None else str.format('S:{0} ', trainsize)
        exp_msg  = '         ' if exp_rate is None else str.format('Exp:{0:.2f} ', exp_rate)
        dist_msg = ''
        if distribution is not None:
            dist_msg = 'dist: '
            for i in range(self.num_class):
               dist_msg = dist_msg + str.format('{0:3.0f} ', distribution[i])

        msg = epi_msg + rewd_msg + size_msg + exp_msg + dist_msg
        print(msg)

        # File Log
        msg = [episode, reward, trainsize, exp_rate, '']
        msg.extend(distribution)

        if low_reward is not None:
            msg.append('')
            msg.append(low_reward)

        if low_distribution is not None:
            msg.append('')
            msg.extend(low_distribution)

        self.logger.log(msg)


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


