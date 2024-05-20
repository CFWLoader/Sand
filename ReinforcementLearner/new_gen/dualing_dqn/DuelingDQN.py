import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as nnf


class DuelingNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, nl1_hidden=10):
        super(DuelingNetwork, self).__init__()
        self.fnn1 = nn.Linear(n_inputs, nl1_hidden).cuda()
        self.fnn1.weight.data.normal_(0, 0.3)

        self.value_layer = nn.Linear(nl1_hidden, n_outputs).cuda()
        self.value_layer.weight.data.normal_(0, 0.3)
        self.advantage_layer = nn.Linear(nl1_hidden, n_outputs).cuda()
        self.advantage_layer.weight.data.normal_(0, 0.3)

    def forward(self, x):
        l1x = self.fnn1(x)
        l1x_rl = nnf.relu(l1x)
        val = self.value_layer(l1x_rl)
        adv = self.advantage_layer(l1x_rl)
        # 合并 Val 和 Adv, 为了不让 Adv 直接学成了 Q, 我们减掉了 Adv 的均值
        adv_mean = adv.mean(dim=1, keepdim=True)
        out = val + (adv - adv_mean)
        return out


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            num_perceptron_hidden=10,
            batch_size=32,
            e_greedy_increment=None,
            dueling=True
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling= dueling
        self.learn_step_counter = 0
        self.num_perceptron_hidden = num_perceptron_hidden
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.loss_hist = []
        self.build_network()

    # 建立神经网络
    def build_network(self):
        self.eval_net = self.build_layers()
        self.target_net = self.build_layers()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        print(self.eval_net)
        print(self.target_net)


    def build_layers(self):
        if self.dueling:
            return DuelingNetwork(self.n_features, self.n_actions)

        fnn1 = nn.Linear(self.n_features, self.num_perceptron_hidden).cuda()
        fnn1.weight.data.normal_(0, 0.3)
        outnet = nn.Linear(self.num_perceptron_hidden, self.n_actions).cuda()
        outnet.weight.data.normal_(0, 0.3)
        return torch.nn.Sequential(
            fnn1,
            nn.ReLU(),
            outnet
        )


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0).cuda()

        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(observation).cpu()
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter & self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(batch_memory[:, :self.n_features]).cuda()
        b_a = torch.LongTensor(batch_memory[:, self.n_features : self.n_features + 1].astype(int)).cuda()
        b_r = torch.FloatTensor(batch_memory[:, self.n_features + 1 : self.n_features + 2]).cuda()
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:]).cuda()

        q_eval = self.eval_net.forward(b_s).gather(1, b_a)
        q_next = self.target_net.forward(b_s_).detach()
        q_next_max = q_next.max(1)
        q_target = b_r + self.gamma * q_next_max[0].unsqueeze(1)
        loss_val = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

        self.loss_hist.append(loss_val.cpu().data.numpy())

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_hist)), self.loss_hist)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.show()
