import numpy as np
import torch
import torch.nn as nn

class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            num_perceptron_hidden=10,
            batch_size=32,
            e_greedy_increment=None,
            double_q=True,
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
        self.double_q = double_q
        self.learn_step_counter = 0
        self.num_perceptron_hidden = num_perceptron_hidden
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.loss_hist = []
        self.build_network()


    def build_network(self):
        eval_fc1 = nn.Linear(self.n_features, self.num_perceptron_hidden).cuda()
        eval_fc1.weight.data.normal_(0, 0.3)
        eval_fc1.bias.data.normal_(0, 0.1)
        eval_out = nn.Linear(self.num_perceptron_hidden, self.n_actions).cuda()
        eval_out.weight.data.normal_(0, 0.3)
        eval_out.bias.data.normal_(0, 0.1)
        self.eval_net = torch.nn.Sequential(
            eval_fc1,
            nn.ReLU(),
            eval_out
        )
        target_fc1 = nn.Linear(self.n_features, self.num_perceptron_hidden).cuda()
        target_fc1.weight.data.normal_(0, 0.3)
        target_fc1.bias.data.normal_(0, 0.1)
        target_out = nn.Linear(self.num_perceptron_hidden, self.n_actions).cuda()
        target_out.weight.data.normal_(0, 0.3)
        target_out.bias.data.normal_(0, 0.1)
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

        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0

        # if np.random.uniform() < self.epsilon:
        action_value = self.eval_net.forward(observation).cpu()
        action = torch.argmax(action_value, 1).data.numpy()[0]

        self.running_q = self.running_q * 0.9 + 0.01 * torch.max(action_value, 1)[1].data.numpy()[0]

        if np.random.uniform() >= self.epsilon:
            action = np.random.randint(0, self.n_actions)

        self.q.append(self.running_q)

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

        if self.double_q:
            q_eval4next = self.eval_net.forward(b_s_).detach()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            max_act4next = q_eval4next.argmax(1).cpu()
            selected_q_next = q_next[batch_index, max_act4next]
            # selected_q_next = q_eval4next.argmax(1)
        else:
            selected_q_next = q_next.max(1)[0]
        q_target = b_r + self.gamma * selected_q_next.unsqueeze(1)

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