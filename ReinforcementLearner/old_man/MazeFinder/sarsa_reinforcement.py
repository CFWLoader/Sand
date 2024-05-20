import pandas as pd

from old_man.MazeFinder.reinforcement_learner import ReinforcementLearner
import numpy as np


class SarsaLearner(ReinforcementLearner):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaLearner, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, cur_state, cur_action, reward, next_state, next_action):
        self.ensure_state(next_state)
        cur_reward = self.exp_table.loc[cur_state, cur_action]
        if next_state != 'terminal':
            target_reward = reward + self.rd * self.exp_table.loc[next_state, next_action]
        else:
            target_reward = reward
        self.exp_table.loc[cur_state, cur_action] += self.lr * (target_reward - cur_reward)


class SarsaLambdaLeaner(ReinforcementLearner):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaLeaner, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambda_val = trace_decay
        self.eligibility_trace = self.exp_table.copy()

    def ensure_state(self, check_state):
        if check_state not in self.exp_table.index:
            pending = pd.Series(np.zeros(len(self.actions)), index=self.exp_table.columns, name=check_state)
            self.exp_table.loc[check_state, :] = pending
            self.eligibility_trace.loc[check_state, :] = pending

    def learn(self, cur_state, cur_action, reward, next_state, next_action):
        self.ensure_state(next_state)
        q_predict = self.exp_table.loc[cur_state, cur_action]
        if next_state != 'terminal':
            q_target = reward + self.rd * self.exp_table.loc[next_state, next_action]
        else:
            q_target = reward
        error_val = q_target - q_predict
        # 这里开始不同:
        # 对于经历过的 state-action, 我们让他+1, 证明他是得到 reward 路途中不可或缺的一环
        self.eligibility_trace.loc[cur_state, cur_action] += 1
        # Q table 更新
        self.exp_table += self.lr * error_val * self.eligibility_trace
        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.lr * self.lambda_val

