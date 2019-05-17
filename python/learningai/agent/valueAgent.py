
from learningai.agent.model.mnist64x5 import mnist64x5_model

class valueAgent(mnist64x5_model):
    def __init__(self, sess, env, lr=1e-3, gamma=0.9):

        print("Create a State Value Estimation agent!")
        self.sess = sess
        self.env = env
        super(valueAgent, self).__init__(n_class=self.env.nclass, lr=1e-3, gamma=0.9, scopeName="dqn_mnist64")

        