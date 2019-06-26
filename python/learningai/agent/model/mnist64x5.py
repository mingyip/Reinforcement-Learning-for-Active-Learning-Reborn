import tensorflow as tf

class mnist64x5_model(object):
    def __init__(self, n_class, lr=1e-3, gamma=0.9, scopeName="mnist64"):
        """
        Build a new model to be used in agent.py for training
        """

        with tf.variable_scope(scopeName):
            # Placeholder Input
            s = tf.placeholder(tf.float32, [None, n_class+1024+2])
            # b = tf.placeholder(tf.float32, [None, 1])
            # e = tf.placeholder(tf.float32, [None, 1])
            R = tf.placeholder(tf.float32, [None, 1])
            avg_V_ = tf.placeholder(tf.float32, [None, 1])

             # Network
            # concat = tf.concat([s, b, e], 1)
            # fc1 = tf.layers.dense(concat, 64, tf.nn.relu)
            fc1 = tf.layers.dense(s, 128, tf.nn.relu)
            fc2 = tf.layers.dense(fc1, 32, tf.nn.relu)
            fc3 = tf.layers.dense(fc2, 16, tf.nn.relu)
            fc4 = tf.layers.dense(fc3, 8, tf.nn.relu)
            V = tf.layers.dense(fc4, 1, tf.nn.relu)
            avg_V = tf.reduce_mean(V)

            # Optimizer
            td_error = R + gamma * avg_V_ - avg_V
            loss = tf.square(td_error)
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss)

            # Variables
            self.s = s
            # self.b = b
            # self.e = e
            self.R = R
            self.V = V
            self.loss = loss
            self.avg_V = avg_V
            self.avg_V_ = avg_V_
            self.td_error = td_error
            self.train_op = train_op
            self.optimizer = optimizer