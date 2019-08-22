
import numpy as np


from config import Config


class AgentLogger(object):

    @staticmethod
    def log_evaluation_results(logs, logger, best_episode, best_reward, mean):
        msg = []
        msg.append(["Evaluate The Best Network with Test data:", "Episode", best_episode, "Accuracy", best_reward])
        logger.log(msg, newline=True)
        AgentLogger.log_training_results(logs, logger)
        logger.log([["Mean", mean]])

    @staticmethod
    def log_training_init(logger):
        msg = []
        msg.append(["Episode", "Accuracy", "Train size", "Exporation Rate", 
                    "Top_dist", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                    "Top_pred", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "", "low Accuracy", 
                    "low_dist", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                    "low_pred", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ])
        logger.log(msg, newline=True)

    @staticmethod
    def print_trianing_results(logs):
        """ Print training logs to console """

        if type(logs) is dict:
            logs = [logs]

        for log in logs:

            # Console Log
            epi_msg  = '        ' if log["episode"] is None else str.format('Eps:{0:3.0f} ', log["episode"])    
            rewd_msg = '         ' if log["top_reward"] is None else str.format('R:{0:.4f} ', log["top_reward"])
            size_msg = '       ' if log["trainsize"] is None else str.format('S:{0} ', log["trainsize"])
            exp_msg  = '         ' if log["exp_rate"] is None else str.format('Exp:{0:.2f} ', log["exp_rate"])
            dist_msg = ''
            if log["top_dist"] is not None:
                dist_msg = 'dist: '
                for i in range(len(log["top_dist"])):
                    dist_msg = dist_msg + str.format('{0:3.0f} ', log["top_dist"][i])

            msg = epi_msg + rewd_msg + size_msg + exp_msg + dist_msg
            print(msg)

    @staticmethod
    def log_training_results(logs, logger):
        """ Write training logs to file """

        msgs = []
        if type(logs) is dict:
            logs = [logs]

        for log in logs:
            # File Log
            msg = [log["episode"], log["top_reward"], log["trainsize"], log["exp_rate"], '']
            msg.extend(log["top_dist"])

            if "top_pred" in log and log["top_pred"] is not None:
                msg.append('')
                msg.extend(log["top_pred"])

            if "low_reward" in log and log["low_reward"] is not None:
                msg.append('')
                msg.append(log["low_reward"])

            if "low_dist" in log and log["low_dist"] is not None:
                msg.append('')
                msg.extend(log["low_dist"])

            if "low_pred" in log and log["low_pred"] is not None:
                msg.append('')
                msg.extend(log["low_pred"])

            msgs.append(msg)

        logger.log(msgs)

    @staticmethod
    def log_bias_prediction(self, env, agent, train_step=0):
        if train_step == 0:
            msg = ['train_steps', '', 'counts', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                '', 'avg_score', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '', 'top_predict_value', 
                '', 'top_y_count', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '', 'top_value', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            
            self.logger.log(msg, logfile='bias.csv')

        
        # Load Config Setting
        budget          = Config.EVALUATION_CLASSIFICATION_BUDGET
        selection_size  = Config.EVALUATION_SELECTION_BATCHSIZE
        train_size      = Config.EVALUATION_TRAINING_BATCHSIZE
        iterations      = int(budget/train_size)
        S               = np.zeros((selection_size, self.num_class+2))
        remain_episodes = 0
        remain_budget   = 1
        num_eval_imgs   = -1
        distribution    = np.zeros((self.num_class))
        
        # Get Env State
        [x_select, y_select, idx]   = env.get_next_selection_batch(batchsize=selection_size)
        S[:, 0:-2]                  = agent.get_next_state_from_env(x_select)
        S[:, -2]                    = remain_budget
        S[:, -1]                    = remain_episodes
        predicts, tops, lows        = agent.predict(S, batchsize=train_size)
        
        # Count 
        y_labels = np.argmax(y_select, axis=1)
        _, counts = np.unique(y_labels, return_counts=True)
        avg_predicts = [np.average(predicts[y_labels == i]) for i in range(10)]

        _, top_counts = np.unique(np.argmax(y_select[tops], axis=1), return_counts=True)
        top_predict_value = predicts[tops]
        top_predict_digit = np.argmax(y_select[tops], axis=1)
        avg_top_predicts = [np.average(predicts[tops])]
        top_y       = np.argmax(y_select[tops], axis=1)
        top_y_unique, top_y_count = np.unique(top_y, return_counts=True)
        top_predict = predicts[tops]
        top_value   = np.zeros(10)
        # print(top_predict)
        for i in range(10):
            val = top_predict[top_y == i]
            if val.size != 0:
                top_value = []

        msg = [train_step]
        msg.append('')
        msg.append('')
        msg.extend(counts)
        msg.append('')
        msg.append('')
        msg.extend(avg_predicts)
        msg.append('')
        # print(avg_top_predicts)
        # print(top_y)
        # print(top_y_unique)
        # print(top_y_count)
        msg.extend(avg_top_predicts)
        msg.append('')
        msg.append('')

        msg.extend(top_y_count)
        msg.append('')
        msg.append('')
        msg.extend(top_value)
        msg.append('')


        self.logger.log(msg, logfile='bias.csv')

