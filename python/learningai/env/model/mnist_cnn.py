import tensorflow as tf


class mnist_cnn(object):
    def __init__(self, img_w, img_h, n_class, lr=1e-4, scopeName = "cnn"):
        with tf.variable_scope(scopeName):
            # Placeholder Input
            x = tf.placeholder(tf.float32, [None, img_w*img_h])           # (batch, height, width, channel)
            y_ = tf.placeholder(tf.float32, [None, n_class])            # input y
            img = tf.reshape(x, [-1, img_w, img_h, 1])

            # Network
            conv1 = tf.layers.conv2d(img, 32, 5, 1, 'same', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2,)
            conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
            pool2f = tf.reshape(pool2, [-1, 7*7*64])
            fc1 = tf.layers.dense(pool2f, 1024, tf.nn.relu)
            output = tf.layers.dense(fc1, n_class)
            y = tf.contrib.layers.softmax(output)

            # Optimizer
            loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss)

            # Variables
            self.x = x
            self.y = y
            self.y_ = y_
            self.fc1 = fc1
            self.loss = loss
            self.accuracy = accuracy
            self.train_op = train_op
            self.optimizer = optimizer
            self.correct_prediction = correct_prediction