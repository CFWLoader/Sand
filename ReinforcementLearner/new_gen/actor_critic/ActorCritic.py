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
        new_dataform = s[np.newaxis, :]
        probs = self.train_net.forward(new_dataform)
        log_prob = torch.log(probs)
        exp_v = log_prob * td

    def choose_action(self, s):
        # 根据 s 选 行为 a
        pass


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        # 用 tensorflow 建立 Critic 神经网络,
        # 搭建好训练的 Graph.
        pass

    def learn(self, s, r, s_):
        # 学习 状态的价值 (state value), 不是行为的价值 (action value),
        # 计算 TD_error = (r + v_) - v,
        # 用 TD_error 评判这一步的行为有没有带来比平时更好的结果,
        # 可以把它看做 Advantage
        return  # 学习时产生的 TD_error


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        # 用 tensorflow 建立 Critic 神经网络,
        # 搭建好训练的 Graph.
        pass

    def learn(self, s, r, s_):
        # 学习 状态的价值 (state value), 不是行为的价值 (action value),
        # 计算 TD_error = (r + v_) - v,
        # 用 TD_error 评判这一步的行为有没有带来比平时更好的结果,
        # 可以把它看做 Advantage
        return  # 学习时产生的 TD_error
