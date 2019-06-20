import tensorflow as tf
import numpy as np
import copy


from learningai.agent.model.mnist64x5 import mnist64x5_model
from config import Config

class valueAgent(object):
    def __init__(self, sess, env, lr=1e-3, gamma=0.9, num_class=10):

        print("Create a State Value Estimation agent!")
        self.num_class = num_class
        self.sess = sess
        self.env = env
        # super(valueAgent, self).__init__(n_class=self.env.nclass, lr=lr, gamma=gamma, scopeName="dqn_mnist64x5")
        self.dqn_init = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_init")
        self.dqn = mnist64x5_model(self.env.nclass, gamma=gamma, scopeName="dqn_train")
        self.memory = []

    def train(self):
        """ Reinforcement Learning Algorithm """
        # TODO: Fix the second remain_budget and remain_episodes bug.
        # TODO: avg should move the gpu to calulate
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


        for episode in range(episodes):
            self.begin_episode()
            reward = self.get_validation_accuracy(nImages=validation_images)

            for iteration in range(int(budget/train_size)):
    
                if iteration <  train_start_at: continue
                if iteration >= train_end_at:   break

                ntrained        = iteration * train_size
                remain_budget   = (budget - ntrained) / budget
                remain_new_bgt  = max(budget-ntrained-1, 0) / budget
                remain_episodes = (episodes - episode) / episodes
                S_old           = copy.deepcopy(S)
                # reward_old      = reward

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


                self.train_agent(reward, S[train_idx], avg_V)

                print("Eps:", episode, " Iter:", iteration, " Reward:", reward, end="\r")
                # print("episode:", episode, "  iteration:", iteration, "reward: ", reward, "exporation_rate:", exporation_rate)

            [top_reward, top_dist, top_size] = self.evaluate(isStream=True, trainTop=True)
            print(str.format('Eps:{0:3.0f} R:{1:.4f} S:{3} Exp:{2:.2f} ', episode, top_reward, exporation_rate, top_size), end='')
            print(str.format('dist:{0:3.0f} {1:3.0f} {2:3.0f} {3:3.0f} {4:3.0f} {5:3.0f} {6:3.0f} {7:3.0f} {8:3.0f} {9:3.0f}', top_dist[0], top_dist[1], top_dist[2], top_dist[3], top_dist[4], top_dist[5], top_dist[6], top_dist[7], top_dist[8], top_dist[9]))

            # if top_reward > 0.95:
            #     for i in range(10):
            #         [top_reward, top_dist, top_size] = self.evaluate(isStream=True, trainTop=True)
            #         # [low_reward, low_dist, low_size] = self.evaluate(isStream=True, trainTop=False)

            #         # print("Eps:", episode, " R:", reward, " ExpRate:", exporation_rate)
            #         print(str.format('        R:{0:.4f} S:{1}          ', top_reward, top_size), end='')
            #         print(str.format('dist:{0:3.0f} {1:3.0f} {2:3.0f} {3:3.0f} {4:3.0f} {5:3.0f} {6:3.0f} {7:3.0f} {8:3.0f} {9:3.0f}', top_dist[0], top_dist[1], top_dist[2], top_dist[3], top_dist[4], top_dist[5], top_dist[6], top_dist[7], top_dist[8], top_dist[9]))
            #         # print(str.format('        R:{0:.2f} S:{1}          ', low_reward, low_size), end='')
            #         # print(str.format('dist:{0:3.`0f} {1:3.0f} {2:3.0f} {3:3.0f} {4:3.0f} {5:3.0f} {6:3.0f} {7:3.0f} {8:3.0f} {9:3.0f}', low_dist[0], low_dist[1], low_dist[2], low_dist[3], low_dist[4], low_dist[5], low_dist[6], low_dist[7], low_dist[8], low_dist[9]))

            if exporation_rate > 0:
                exporation_rate -= exporation_decay_rate


    def begin_game(self):
        """ Reset the agent in the game """
        self.resetNetwork()
        self.env.resetEnvCounter()

    def begin_episode(self):
        """ Reset the agent memory and the environment """
        # Done: reset the agent memory
        self.env.resetNetwork()
        self.memory = []

    def reset_network(self):
        """ Reset the classification network """
        # TODO: maybe merge with the function begin_episode
        self.env.resetNetwork()

    def get_next_state_from_env(self, imgs):
        """ Get output probability of new images from the environment """
        state = self.env.get_output_probability(imgs)
        return state

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

    def evaluate(self, isStream=True, trainTop=True):
        """ Agent uses the current policy to train a new network """
        """ Both stream and pool based learning """
        budget          = Config.CLASSIFICATION_BUDGET
        episodes        = Config.AGENT_TRAINING_EPISODES
        epochs          = Config.CLASSIFICATION_EPOCH
        selection_size  = Config.SELECTION_BATCHSIZE
        train_size      = Config.TRAINING_BATCHSIZE
        iterations      = int(budget/train_size)
        S               = np.zeros((selection_size, self.num_class+2))
        remain_episodes = 0
        validation_imgs = -1
        distribution    = np.zeros((10))
        trainSize       = 0

        self.reset_network()
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

        return [self.get_test_accuracy(validation_imgs), distribution, trainSize]

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
