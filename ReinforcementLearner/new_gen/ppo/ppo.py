import numpy as np
from torch import nn, optim, tensor, normal


class PPOActorOutLayer(nn.Module):
    def __init__(self, num_l1_hidden, a_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_net = nn.Linear(num_l1_hidden, a_dim)
        self.sigma_net = nn.Linear(num_l1_hidden, a_dim)

    def forward(self, s: tensor) -> tensor:
        mu = self.mu_net.forward(2 * s)
        sigma = self.sigma_net.forward(s)
        return normal(mu.tanh(), nn.functional.softplus(sigma))


class ProximalPolicyOptimization:
    def __init__(self,
                 a_dim,
                 s_dim,
                 a_bound,
                 lr_actor=0.0001,
                 lr_critic=0.0002,
                 reward_discount=0.9,
                 batch_size=32,
                 update_steps_actor=10,
                 update_steps_critic=10,
                 actor_l1_hidden=100,
                 critic_l1_hidden=100
                 ):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.critic_net = self.build_critic(self.s_dim, critic_l1_hidden)
        self.pi_net = self.build_actor(self.s_dim, self.a_dim, actor_l1_hidden)
        self.old_pi_net = self.build_actor(self.s_dim, self.a_dim, actor_l1_hidden)

    def update(self, s, a, r):
        pass

    def choose_action(self, s):
        s_input = tensor(s[np.newaxis, :]).cuda()
        norm_dist: tensor = self.pi_net.forward(s_input).detach()
        norm_dist
        # a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        # return np.clip(a, -2, 2)
        pass

    def get_v(self, s):
        # 算 state value
        # if s.ndim < 2: s = s[np.newaxis, :]
        # return self.sess.run(self.v, {self.tfs: s})[0, 0]
        pass

    @staticmethod
    def build_actor(s_dim, a_dim, num_l1_hidden):
        l1 = nn.Linear(s_dim, num_l1_hidden)
        norm_layer = PPOActorOutLayer(num_l1_hidden, a_dim)
        return nn.Sequential(l1, nn.ReLU(), norm_layer)

    @staticmethod
    def build_critic(s_dim, num_l1_hidden):
        l1 = nn.Linear(s_dim, num_l1_hidden)
        cr = nn.Linear(num_l1_hidden, 1)
        return nn.Sequential(l1, cr)
