import numpy as np
import torch
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


class DDPGCritic(nn.Module):

    def __init__(self, s_dim, a_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.ws = torch.ones(s_dim, out_dim, requires_grad=True).cuda()
        self.wa = torch.ones(a_dim, out_dim, requires_grad=True).cuda()
        self.b1 = torch.ones(out_dim, requires_grad=True).cuda()
        self.wsp = nn.Parameter(self.ws)
        self.wap = nn.Parameter(self.wa)
        self.b1p = nn.Parameter(self.b1)
        self.register_parameter('state_weights', self.wsp)
        self.register_parameter('action_weights', self.wap)
        self.register_parameter('sa_bias', self.b1p)

    def forward(self, in_state, in_action):
        ten_s = torch.Tensor(in_state).cuda()
        ten_a = torch.Tensor(in_action).cuda()
        sm = torch.matmul(ten_s, self.ws)
        am = torch.matmul(ten_a, self.wa)
        return (sm + am + self.b1).relu()


class DeepDeterministicPolicyGradient(object):
    def __init__(self, a_dim, s_dim, a_bound, **kwargs):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.l1hidden_n = 30
        # Actor
        self.eval_actor = self.build_actor(self.s_dim, self.a_dim, self.l1hidden_n)
        self.target_actor = self.build_actor(self.s_dim, self.a_dim, self.l1hidden_n)
        # Critic
        self.eval_critic = DDPGCritic(self.s_dim, self.a_dim, self.l1hidden_n)
        self.target_critic = DDPGCritic(self.s_dim, self.a_dim, self.l1hidden_n)
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

    def build_actor(self, n_features, n_actions, l1units=30):
        layer1 = nn.Linear(n_features, l1units)
        act_layer = nn.Linear(l1units, n_actions)
        return nn.Sequential(layer1, nn.ReLU(), act_layer, nn.Tanh())

    def build_critic(self, n_features, n_actions, l1units=30):
        pass


###############################  training  ####################################