import torch.nn
import torch.nn as tnn
import numpy as np


class PolicyGradient:
    # 初始化 (有改变)
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # reward 递减率
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # 这是我们存储 回合信息的 list
        self._build_net()  # 建立 policy 神经网络

    # 建立 policy gradient 神经网络 (有改变)
    def _build_net(self):
        n_hidden = 10
        fc1 = tnn.Linear(self.n_features, n_hidden).cuda()
        fc1.weight.data.normal_(0, 0.3)
        # fc1.bias.data.constant_(0.1)
        tnn.init.constant_(fc1.bias.data, 0.1)

        all_act = tnn.Linear(n_hidden, self.n_actions).cuda()
        all_act.weight.data.normal_(0, 0.3)
        tnn.init.constant_(all_act.weight.data, .01)
        # all_act.bias.data.constant_(0.1)

        self.neural_net = tnn.Sequential(
            fc1,
            tnn.Tanh(),
            all_act,
            tnn.Softmax()
        )

        self.loss_func = tnn.CrossEntropyLoss(reduce=False)
        # self.LogSoftMax = tnn.LogSoftmax(dim=-1)
        # self.loss_func2 = tnn.NLLLoss(reduce=False)
        self.optim = torch.optim.Adam(self.neural_net.parameters(), lr=self.lr)
        print(self.neural_net)

    # 选行为 (有改变)
    def choose_action(self, observation):
        prob_weights_res = self.neural_net.forward(torch.Tensor(observation).cuda()).cpu()
        prob_weights = prob_weights_res.detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights.ravel())
        return action

    # 存储回合 transition (有改变)
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # 学习更新参数 (有改变)
    # def learn(self):
    #     # discount and normalize episode reward
    #     discounted_ep_rs_norm = self._discount_and_norm_rewards()
    #     self.tf_vt = discounted_ep_rs_norm
    #
    #     '''
    #     algo1
    #     这段是模拟tensorflow.nn.sparse_softmax_cross_entropy_with_logit行为，是优化后的做法
    #     ptlbls = torch.tensor([]).int()
    #     ptlgts = torch.tensor([])
    #     diff1 = self.loss_func1(ptlbls, ptlgts)
    #     diffLSM = self.LogSoftMax(diff1)
    #     neg_log_prob = self.loss_func2(diffLSM, ptlgts.long())
    #     loss_val = neg_log_prob * self.tf_vt
    #     '''
    #     # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
    #     # loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
    #     self.all_act = self.neural_net.forward(s)
    #     self.all_act_prob = tnn.Softmax(self.all_act)
    #     act2one_hot = torch.nn.functional.one_hot(self.all_act, self.n_actions)
    #     neg_log_prob = -torch.log(self.all_act_prob) * act2one_hot
    #     loss_val = None
    #
    #     self.optim.zero_grad()
    #     loss_val.backward()
    #     self.optim.step()
    #
    #     self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
    #     return discounted_ep_rs_norm

    # 学习更新参数 (有改变)
    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.torch_vt = torch.Tensor(discounted_ep_rs_norm).cuda()
        self.torch_acts = torch.Tensor(np.array(self.ep_as)).long()
        self.torch_obs = torch.Tensor(np.vstack(self.ep_obs))
        self.act2one_hot = torch.nn.functional.one_hot(self.torch_acts, self.n_actions).float()

        self.all_act = self.neural_net.forward(self.torch_obs.cuda())
        self.all_act_prob = tnn.Softmax(self.all_act)

        neg_log_prob = self.loss_func(self.all_act, self.act2one_hot.cuda())

        loss_val = torch.mean(neg_log_prob * self.torch_vt)

        self.optim.zero_grad()
        loss_val.backward()
        self.optim.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    # 衰减回合的 reward (新内容)
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
