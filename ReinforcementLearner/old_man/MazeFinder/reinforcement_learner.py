import numpy as np
import pandas as pd


class ReinforcementLearner:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.rd = reward_decay
        self.eps = e_greedy
        self.exp_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, src_state):
        self.ensure_state(src_state)
        if np.random.uniform() < self.eps:
            state_actions = self.exp_table.loc[src_state, :]
            return np.random.choice(state_actions[state_actions == state_actions.max()].index)
        else:
            return np.random.choice(self.actions)

    def learn(self, *args):
        pass

    def ensure_state(self, check_state):
        if check_state not in self.exp_table.index:
            self.exp_table.loc[check_state, :] = np.zeros(len(self.actions))

    def export_weights(self, csv_path='sarsa_learner.csv'):
        print('exporting %s' % csv_path)
        self.exp_table.to_csv(csv_path)
