import numpy as np
from torch import nn, optim, tensor, normal, distributions


class PPOActorOutLayer(nn.Module):
    def __init__(self, num_l1_hidden, a_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_net = nn.Linear(num_l1_hidden, a_dim)
        self.sigma_net = nn.Linear(num_l1_hidden, a_dim)

    def forward(self, s: tensor) -> distributions.normal.Normal:
        mu = self.mu_net.forward(2 * s)
        sigma = self.sigma_net.forward(s)
        nordist = distributions.normal.Normal(mu, nn.functional.softplus(sigma))
        return nordist
        # return normal(mu.tanh(), nn.functional.softplus(sigma))


class DivergenceSmoothMethodConfig:
    def __init__(self, name):
        self.name = name
        # KL penalty
        self.kl_target = 0.01
        self.lam = 0.5
        # clipped surrogate objective
        self.epsilon = 0.2


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
                 critic_l1_hidden=100,
                 dsmc=DivergenceSmoothMethodConfig('kl_pen')
                 ):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.critic_net = self.build_critic(self.s_dim, critic_l1_hidden)
        self.pi_net = self.build_actor(self.s_dim, self.a_dim, actor_l1_hidden)
        self.old_pi_net = self.build_actor(self.s_dim, self.a_dim, actor_l1_hidden)
        self.critic_train = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        self.actor_train = optim.Adam(self.pi_net.parameters(), lr=lr_actor)
        self.dsmc = dsmc

    def update(self, s, a, r):
        self.overwrite_old_net()
        input_state = tensor(s).cuda()
        took_action = tensor(a).cuda()
        discounted_rw = tensor(r).cuda()
        # update critic
        adv = self.update_critic(input_state, discounted_rw)

    def update_critic(self, in_state: tensor, discounted_rw: tensor) -> tensor:
        critic_out = self.critic_net.forward(in_state)
        advantages = discounted_rw - critic_out
        critic_loss = advantages.square()
        self.critic_train.zero_grad()
        critic_loss.backward()
        self.critic_train.step()
        return advantages.detach()

    def update_actor(self, in_state: tensor, in_action: tensor, adv_val: tensor):
        pass

    def choose_action(self, s):
        s_input = tensor(s[np.newaxis, :]).cuda()
        norm_dist: distributions.normal.Normal = self.pi_net.forward(s_input).detach()
        action_val = np.squeeze(norm_dist.sample().detach().cpu(), axis=0)
        # a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(action_val, -2, 2)

    def get_v(self, s):
        # 算 state value
        # if s.ndim < 2: s = s[np.newaxis, :]
        # return self.sess.run(self.v, {self.tfs: s})[0, 0]
        pass

    def overwrite_old_net(self):
        self.old_pi_net.load_state_dict(self.pi_net.state_dict())

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
