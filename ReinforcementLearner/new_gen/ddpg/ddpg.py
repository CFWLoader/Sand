import numpy as np
from torch import nn, optim, Tensor, hstack

MAX_EPISODES = 200
MAX_EP_STEPS = 200
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

###############################  DDPG  ####################################


# class DDPGCritic(nn.Module):
#
#     def __init__(self, s_dim, a_dim, out_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.ws = torch.ones(s_dim, out_dim, requires_grad=True).cuda()
#         # self.wa = torch.ones(a_dim, out_dim, requires_grad=True).cuda()
#         self.b1 = torch.ones(out_dim, requires_grad=True).cuda()
#         nn.init.normal_(self.ws, 0, 0.1)
#         # nn.init.normal_(self.wa, 0, 0.1)
#         nn.init.constant_(self.b1, 0.1)
#         self.wsp = nn.Parameter(self.ws)
#         # self.wap = nn.Parameter(self.wa)
#         self.b1p = nn.Parameter(self.b1)
#         self.register_parameter('state_weights', self.wsp)
#         # self.register_parameter('action_weights', self.wap)
#         self.register_parameter('sa_bias', self.b1p)
#
#     def forward(self, in_state, in_action):
#         ten_s = torch.Tensor(in_state).cuda()
#         ten_a = torch.Tensor(in_action).cuda()
#         sm = torch.matmul(ten_s, self.ws)
#         # am = torch.matmul(ten_a, self.wa)
#         # return (sm + am + self.b1).relu()
#         return (sm + self.b1).relu()


class DeepDeterministicPolicyGradient(object):
    def __init__(self, a_dim, s_dim, a_bound, lr_actor=0.001, lr_critic=0.002, **kwargs):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.l1hidden_n = 30
        # Actor
        self.eval_actor = self.build_actor(self.s_dim, self.a_dim, self.l1hidden_n)
        self.target_actor = self.build_actor(self.s_dim, self.a_dim, self.l1hidden_n)
        self.actor_train = optim.Adam(self.eval_actor.parameters(), lr=lr_actor)
        # Critic
        # self.eval_critic = DDPGCritic(self.s_dim, self.a_dim, self.l1hidden_n)
        # self.target_critic = DDPGCritic(self.s_dim, self.a_dim, self.l1hidden_n)
        self.eval_critic = self.build_critic(self.s_dim, self.a_dim, self.l1hidden_n)
        self.target_critic = self.build_critic(self.s_dim, self.a_dim, self.l1hidden_n)
        self.critic_train = optim.Adam(self.eval_critic.parameters(), lr=lr_critic)
        # ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

    def choose_action(self, s):
        fit_form = Tensor(s[np.newaxis, :]).cuda()
        predict_data = self.eval_actor.forward(fit_form).detach().cpu()
        return predict_data.to_numpy()

    def learn(self):
        # @ToBeFill actor/critic target的参数更新，soft的方法为衰变+eval的值，hard为直接替换
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = Tensor(bt[:, :self.s_dim]).cuda()
        ba = Tensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).cuda()
        br = Tensor(bt[:, -self.s_dim - 1: -self.s_dim]).cuda()
        bs_ = Tensor(bt[:, -self.s_dim:]).cuda()
        # train actor
        eval_act = self.eval_actor.forward(bs)
        evact_det = eval_act.detach()
        eval_qval = self.eval_critic.forward(hstack((bs, evact_det)))
        actor_loss = -eval_qval.sum()
        self.actor_train.zero_grad()
        actor_loss.backward()
        self.actor_train.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def build_actor(self, n_features, n_actions, l1units=30):
        layer1 = nn.Linear(n_features, l1units).cuda()
        nn.init.normal_(layer1.weight.data, 0, 0.1)
        nn.init.constant_(layer1.bias.data, 0.1)
        act_layer = nn.Linear(l1units, n_actions).cuda()
        nn.init.normal_(act_layer.weight.data, 0, 0.1)
        nn.init.constant_(act_layer.bias.data, 0.1)
        return nn.Sequential(layer1, nn.ReLU(), act_layer, nn.Tanh())

    def build_critic(self, n_features, n_actions, l1units=30):
        layer1 = nn.Linear(n_features + n_actions, l1units).cuda()
        nn.init.normal_(layer1.weight.data, 0, 0.1)
        nn.init.constant_(layer1.bias.data, 0.1)
        q_score = nn.Linear(l1units, 1).cuda()
        nn.init.normal_(q_score.weight.data, 0, 0.1)
        nn.init.constant_(q_score.bias.data, 0.1)
        return nn.Sequential(layer1, nn.ReLU(), q_score)
        # layer1 = nn.Linear(n_features + n_actions, l1units).cuda()
        # nn.init.constant_(layer1.weight.data, 1)
        # nn.init.constant_(layer1.bias.data, 1)
        # return layer1


###############################  training  ####################################