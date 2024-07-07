import numpy as np
from torch import nn, optim

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

###############################  DDPG  ####################################


class DeepDeterministicPolicyGradient(object):
    def __init__(self, a_dim, s_dim, a_bound, **kwargs):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        # ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

    def choose_action(self, s):
        pass

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def build_actor(self, n_features, n_actions, l1units=30, reuse=None, custom_getter=None):
        layer1 = nn.Linear(n_features, l1units)
        act_layer = nn.Linear(l1units, n_actions)
        return nn.Sequential(layer1, nn.ReLU(), act_layer, nn.Tanh())

    def build_critic(self, n_features, n_actions, l1units=30, reuse=None, custom_getter=None):
        pass


###############################  training  ####################################