import enum

import numpy as np
import torch
from torch import nn, optim, tensor, normal, distributions


class PPOActorOutLayer(nn.Module):
    def __init__(self, num_l1_hidden, a_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_net = nn.Linear(num_l1_hidden, a_dim)
        self.sigma_net = nn.Linear(num_l1_hidden, a_dim)

    def forward(self, s: tensor) -> tensor:
        mu = self.mu_net.forward(2 * s)
        sigma = self.sigma_net.forward(s)
        nordist = distributions.normal.Normal(mu, nn.functional.softplus(sigma))
        return nordist.sample(torch.Size((1,)))
        # return normal(mu.tanh(), nn.functional.softplus(sigma))


class ProximalActor(nn.Module):
    def __init__(self, s_dim, a_dim, num_l1_hidden, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(s_dim, num_l1_hidden).cuda()
        self.mu_net = nn.Linear(num_l1_hidden, a_dim).cuda()
        self.sigma_net = nn.Linear(num_l1_hidden, a_dim).cuda()
        # return nn.Sequential(l1, nn.ReLU(), norm_layer)
        self.last_mu = None
        self.last_sigma = None

    def forward(self, s: tensor) -> tensor:
        l1out = self.l1.forward(s)
        mu = self.mu_net.forward(2 * l1out)
        sigma = self.sigma_net.forward(l1out)
        self.last_mu = mu.detach()
        self.last_sigma = sigma.detach()
        nordist = distributions.normal.Normal(mu, nn.functional.softplus(sigma))
        return nordist.rsample(torch.Size((1,)))
        # return nordist.sample(torch.Size((1,)))

    def get_distribution(self):
        return distributions.normal.Normal(self.last_mu, nn.functional.softplus(self.last_sigma))


class PenaltyMethodName(enum.Enum):
    KL_PENALTY = 1
    CLIP = 2


class DivergenceSmoothMethodConfig:
    def __init__(self, name: PenaltyMethodName):
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
                 lr_actor=0.01,
                 lr_critic=0.02,
                 reward_discount=0.9,
                 batch_size=32,
                 update_steps_actor=10,
                 update_steps_critic=10,
                 actor_l1_hidden=100,
                 critic_l1_hidden=100,
                 dsmc=DivergenceSmoothMethodConfig(PenaltyMethodName.KL_PENALTY)
                 ):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.update_steps_actor = update_steps_actor
        self.update_steps_critic = update_steps_critic
        self.critic_net = self.build_critic(self.s_dim, critic_l1_hidden)
        self.pi_net = self.build_actor(self.s_dim, self.a_dim, actor_l1_hidden)
        self.old_pi_net = self.build_actor(self.s_dim, self.a_dim, actor_l1_hidden)
        self.critic_loss = nn.MSELoss()
        self.critic_train = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        self.actor_train = optim.Adam(self.pi_net.parameters(), lr=lr_actor)
        self.dsmc = dsmc

    def update(self, s, a, r):
        self.overwrite_old_net()
        input_state = tensor(s, dtype=torch.float32).cuda()
        took_action = tensor(a, dtype=torch.float32).cuda()
        discounted_rw = tensor(r, dtype=torch.float32).cuda()
        # update critic
        adv = self.update_critic(input_state, discounted_rw)
        self.update_actor(input_state, took_action, adv)
        for _ in range(self.update_steps_critic):
            self.update_critic(input_state, discounted_rw)

    def update_critic(self, in_state: tensor, discounted_rw: tensor) -> tensor:
        critic_out: tensor = self.critic_net.forward(in_state)
        advantages = discounted_rw - critic_out
        critic_loss = self.critic_loss(discounted_rw, critic_out) # advantages.square()
        self.critic_train.zero_grad()
        critic_loss.backward()
        self.critic_train.step()
        return advantages.detach()

    def update_actor(self, in_state: tensor, in_action: tensor, adv_val: tensor):
        if self.dsmc.name == PenaltyMethodName.KL_PENALTY:
            kl_mean = self.dsmc.kl_target
            for _ in range(self.update_steps_actor):
                kl_mean = self.update_actor_net(in_state, in_action, adv_val)
                if kl_mean > 4 * self.dsmc.kl_target:
                    break
            if kl_mean < self.dsmc.kl_target / 1.5:
                self.dsmc.lam /= 2
            elif kl_mean > self.dsmc.kl_target * 1.5:
                self.dsmc.lam *= 2
            self.dsmc.lam = np.clip(self.dsmc.lam, 1e-4, 10)
        elif self.dsmc.name == PenaltyMethodName.CLIP:
            for _ in range(self.update_steps_actor):
                self.update_actor_net(in_state, in_action, adv_val)
        else:
            pass

    def update_actor_net(self, in_state: tensor, in_action: tensor, adv_val: tensor):
        pi_val: tensor = self.pi_net.forward(in_state)
        old_pi_val: tensor = self.old_pi_net.forward(in_state).detach()
        pi_dist = self.pi_net.get_distribution()
        old_pi_dist = self.old_pi_net.get_distribution()
        ratio: tensor = torch.exp(pi_dist.log_prob(in_action)) / (torch.exp(old_pi_dist.log_prob(in_action)) + 1e-5)
        surrogate = ratio * adv_val
        return_pack = None
        if self.dsmc.name == PenaltyMethodName.KL_PENALTY:
            torchlam = self.dsmc.lam
            # @TODO 这里的计算方式是否完全对得上
            old_pi_cat = distributions.Categorical(old_pi_val)
            pi_cat = distributions.Categorical(pi_val)
            kl_pen = distributions.kl_divergence(old_pi_cat, pi_cat)
            kl_mean = kl_pen.mean()
            actor_loss = -((surrogate - torchlam * kl_pen).mean())
            return_pack = kl_mean.detach()
        elif self.dsmc.name == PenaltyMethodName.CLIP:
            clipped = ratio.clamp(1. - self.dsmc.epsilon, 1. + self.dsmc.epsilon) * adv_val
            actor_loss = -torch.min(surrogate, clipped).mean()
        else:
            actor_loss = None
        self.actor_train.zero_grad()
        actor_loss.backward()
        self.actor_train.step()
        return return_pack

    def choose_action(self, s):
        # s_input = tensor(s[np.newaxis, :]).cuda()
        s_input = tensor(s).cuda()
        sam_res: tensor = self.pi_net.forward(s_input).detach()
        action_val = np.squeeze(sam_res.detach().cpu().numpy(), axis=0)
        # a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(action_val, -2, 2)

    def get_v(self, s):
        # 算 state value
        # in_state = tensor(s[np.newaxis, :]).cuda() if s.ndim < 2 else tensor(s).cuda()
        in_state = tensor(s[np.newaxis, :]).cuda() if s.ndim < 2 else tensor(s).cuda()
        run_result = self.critic_net.forward(in_state)
        return run_result.detach().cpu().numpy()[0, 0]

    def overwrite_old_net(self):
        self.old_pi_net.load_state_dict(self.pi_net.state_dict())
        self.old_pi_net.last_sigma = self.pi_net.last_sigma.detach()
        self.old_pi_net.last_mu = self.pi_net.last_mu.detach()

    @staticmethod
    def build_actor(s_dim, a_dim, num_l1_hidden) -> ProximalActor:
        return ProximalActor(s_dim, a_dim, num_l1_hidden)
        # l1 = nn.Linear(s_dim, num_l1_hidden)
        # norm_layer = PPOActorOutLayer(num_l1_hidden, a_dim)
        # return nn.Sequential(l1, nn.ReLU(), norm_layer)

    @staticmethod
    def build_critic(s_dim, num_l1_hidden):
        l1 = nn.Linear(s_dim, num_l1_hidden).cuda()
        cr = nn.Linear(num_l1_hidden, 1).cuda()
        return nn.Sequential(l1, cr)
