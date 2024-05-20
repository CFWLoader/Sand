import torch.nn
import torch.nn as tnn

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
        fc1.bias.data.constant_(0.1)
        tnn.init.constant_(fc1.bias.data.constant_, 0.1)

        all_act = tnn.Linear(n_hidden, self.n_actions)
        all_act.weight.data.normal_(0, 0.3)
        all_act.bias.data.constant_(0.1)

        self.neural_net = tnn.Sequential(
            fc1,
            tnn.ReLU(),
            all_act,
            tnn.Softmax()
        )

        self.loss_func1 = tnn.CrossEntropyLoss(reduce=False)
        self.LogSoftMax = tnn.LogSoftmax(dim=-1)
        self.loss_func2 = tnn.NLLLoss(reduce=False)
        self.optim = torch.optim.Adam(self.neural_net.parameters(), lr=self.lr)
        print(self.neural_net)


    # 选行为 (有改变)
    def choose_action(self, observation):
        pass


    # 存储回合 transition (有改变)
    def store_transition(self, s, a, r):
        pass


    # 学习更新参数 (有改变)
    def learn(self, s, a, r, s_):
        ptlbls = torch.tensor([]).int()
        ptlgts = torch.tensor([])
        diff1 = self.loss_func1(ptlbls, ptlgts)
        diffLSM = self.LogSoftMax(diff1)
        neg_log_prob = self.loss_func2(diffLSM, ptlgts.long())
        loss_val = neg_log_prob * self.tf_vt
        self.optim.zero_grad()
        loss_val.backward()
        self.optim.step()


    # 衰减回合的 reward (新内容)
    def _discount_and_norm_rewards(self):
        pass