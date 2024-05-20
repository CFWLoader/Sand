import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as nnf


class DeepQNetwork(object):
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
            output_graph=False
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
        self.learn_step_counter = 0
        self.num_perceptron_hidden = num_perceptron_hidden
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.loss_hist = []
        self.build_network()

    # 建立神经网络
    def build_network(self):
        eval_fc1 = nn.Linear(self.n_features, self.num_perceptron_hidden).cuda()
        eval_fc1.weight.data.normal_(0, 0.1)
        eval_out = nn.Linear(self.num_perceptron_hidden, self.n_actions).cuda()
        eval_out.weight.data.normal_(0, 0.1)
        self.eval_net = torch.nn.Sequential(
            eval_fc1,
            nn.ReLU(),
            eval_out
        )
        target_fc1 = nn.Linear(self.n_features, self.num_perceptron_hidden).cuda()
        target_fc1.weight.data.normal_(0, 0.1)
        target_out = nn.Linear(self.num_perceptron_hidden, self.n_actions).cuda()
        target_out.weight.data.normal_(0, 0.1)
        self.target_net = torch.nn.Sequential(
            target_fc1,
            nn.ReLU(),
            target_out
        )
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        print(self.eval_net)
        print(self.target_net)

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

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.
        # 参照原始算法代码对比思路：
        # cur_reward = self.exp_table.loc[cur_state, cur_action]
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)
        q_next = self.target_net.forward(b_s_).detach()
        # target_reward = reward + self.rd * self.exp_table.loc[next_state, :].max()
        q_next_max = q_next.max(1)
        q_target = b_r + self.gamma * q_next_max[0].unsqueeze(1)
        # self.exp_table.loc[cur_state, cur_action] += self.lr * (target_reward - cur_reward)
        # cur_reward = self.exp_table.loc[cur_state, cur_action]
        loss_val = self.loss_func(q_eval, q_target)

        """
                假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
                q_eval =
                [[1, 2, 3],
                 [4, 5, 6]]

                q_target = q_eval =
                [[1, 2, 3],
                 [4, 5, 6]]

                然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
                比如在:
                    记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
                    记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
                q_target =
                [[-1, 2, 3],
                 [4, 5, -2]]

                所以 (q_target - q_eval) 就变成了:
                [[(-1)-(1), 0, 0],
                 [0, 0, (-2)-(6)]]

                最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
                所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
                我们只反向传递之前选择的 action 的值,
                """

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
