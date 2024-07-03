import torch
from torch import nn, optim, distributions

GAMMA = 0.1


class Actor(nn.Module):
    def __init__(self, n_features, action_bound, lr=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.action_bound = action_bound

        self.l1 = nn.Linear(n_features, 30)  # relu
        self.mu = nn.Linear(30, 1)  # tanh
        self.sigma = nn.Linear(30, 1)  # log(exp(features) + 1)

        self.normal_dist = distributions.Normal(0, 1)
        self.optim = optim.Adam(self.parameters(), lr)

    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=next(self.parameters()).device)[None]
        h = self.l1(s).relu()
        mu = self.mu(h).tanh()
        sigma = self.sigma(h)
        sigma = torch.log(sigma.exp() + 1)

        self.normal_dist = distributions.Normal(mu[0] * 2, sigma[0] + .1)
        action = self.normal_dist.sample()
        action = torch.clip(action, self.action_bound[0], self.action_bound[1])
        return action

    def learn(self, action, td_error):
        action_prob = self.normal_dist.log_prob(action)
        exp_v = action_prob * td_error.detach() + 0.01 * self.normal_dist.entropy()
        loss = -exp_v.sum()

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return exp_v


class Critic(nn.Module):
    def __init__(self, n_features, lr=0.01, **kwargs):
        super().__init__(**kwargs)
        self.l1 = nn.Linear(n_features, 30)  # relu
        self.v = nn.Linear(30, 1)  #
        self.optim = optim.Adam(self.parameters(), lr)

    def forward(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=next(self.parameters()).device)[None]
        return self.v(self.l1(s).relu())

    def learn(self, s, r, s_):
        with torch.no_grad():
            v_ = self(s_)
        td_error = torch.mean((r + GAMMA * v_) - self(s))
        loss = td_error.square()

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return td_error
