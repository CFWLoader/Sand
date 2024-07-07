import torch
from torch import nn, optim
import numpy as np


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        # 搭建好训练的 Graph.
        self.n_hidden_l1 = 20
        self.layer1 = nn.Linear(n_features, self.n_hidden_l1).cuda()
        self.layer1.weight.data.normal_(0, 0.1)
        nn.init.constant_(self.layer1.bias.data, 0.1)

        self.acts_prob = nn.Linear(self.n_hidden_l1, n_actions).cuda()
        self.acts_prob.weight.data.normal_(0, 0.1)
        nn.init.constant_(self.acts_prob.bias.data, 0.1)

        self.train_net = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.acts_prob,
            nn.Softmax()
        )

        self.optim = optim.Adam(self.train_net.parameters(), lr=lr)

    def learn(self, s, a, td):
        # s, a 用于产生 Gradient ascent 的方向,
        # td 来自 Critic, 用于告诉 Actor 这方向对不对.
        new_dataform = torch.Tensor(s[np.newaxis, :]).cuda()
        probs = self.train_net.forward(new_dataform)
        # TF版中是这样访问，应该会有bug
        log_prob = torch.log(probs[0, a])
        exp_v = log_prob * td
        loss_val = -exp_v.sum()
        self.optim.zero_grad()
        loss_val.backward()
        self.optim.step()

    def choose_action(self, s):
        # 根据 s 选 行为 a
        state_df = torch.Tensor(s[np.newaxis, :]).cuda()
        probs = self.train_net.forward(state_df).detach().cpu().numpy()
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):
    def __init__(self, n_features, lr=0.01, td_gamma=0.9):
        """
        @param n_features:
        @param lr:
        @param td_gamma: reward discount in TD error
        """
        # 用 tensorflow 建立 Critic 神经网络,
        # 搭建好训练的 Graph.
        self.td_gamma = td_gamma
        self.n_hidden = 20
        self.layer1 = nn.Linear(n_features, self.n_hidden).cuda()
        self.layer1.weight.data.normal_(.0, .1)
        nn.init.constant_(self.layer1.bias.data, 0.1)

        self.val_layer = nn.Linear(self.n_hidden, 1).cuda()
        self.val_layer.weight.data.normal_(.0, .1)
        nn.init.constant_(self.val_layer.bias.data, 0.1)

        self.train_net = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.val_layer
        )

        self.optim = optim.Adam(self.train_net.parameters(), lr=lr)

    def learn(self, s, r, s_):
        # 学习 状态的价值 (state value), 不是行为的价值 (action value),
        # 计算 TD_error = (r + v_) - v,
        # 用 TD_error 评判这一步的行为有没有带来比平时更好的结果,
        # 可以把它看做 Advantage
        state_np = s[np.newaxis, :]
        state, state_ = torch.Tensor(state_np).cuda(), torch.Tensor(s_[np.newaxis, :]).cuda()
        val_next = self.train_net.forward(state_).detach()
        val_cur = self.train_net.forward(state)
        reward = torch.Tensor(np.full(val_next.shape, r)).cuda()
        td_error = reward + self.td_gamma * val_next - val_cur
        loss_val = td_error.square()
        self.optim.zero_grad()
        loss_val.backward()
        self.optim.step()
        return td_error.detach()
