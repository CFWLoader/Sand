from old_man.MazeFinder.reinforcement_learner import ReinforcementLearner
from game_code.maze_env import Maze
from functools import partial
from old_man.MazeFinder.sarsa_reinforcement import SarsaLearner


class QuantLearner(ReinforcementLearner):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QuantLearner, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, cur_state, cur_action, reward, next_state):
        self.ensure_state(next_state)
        cur_reward = self.exp_table.loc[cur_state, cur_action]
        if next_state == 'terminal':
            target_reward = reward
        else:
            target_reward = reward + self.rd * self.exp_table.loc[next_state, :].max()
        self.exp_table.loc[cur_state, cur_action] += self.lr * (target_reward - cur_reward)


def update(learner):
    for episode in range(100):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = learner.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            learner.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

        print('epoch %d - reward %d' % (episode, reward))
        # if episode % 5 == 0:
        #     learner.export_weights('maze_epoch%d.csv' % episode)
        # print(RL.q_table)

    # end of game
    print('game over')
    env.destroy()


def sarsa_update(learner):
    for episode in range(100):
        # initial observation
        observation = env.reset()

        action = learner.choose_action(str(observation))

        steps = 1

        action_list = [action]

        while True:
            # fresh env
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = learner.choose_action(str(observation_))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            learner.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_

            action = action_

            action_list.append(action)

            steps += 1

            # break while loop when end of this episode
            if done:
                break

        print('epoch %d - reward %d(steps=%d)' % (episode, reward, steps))

        if steps <= 4:
            print(action_list)
        # if episode % 5 == 0:
        #     learner.export_weights('maze_epoch%d.csv' % episode)
        # print(RL.q_table)

    # end of game
    print('game over')
    env.destroy()


def sarsa_lam_update(learner):
    for episode in range(100):
        # initial observation
        observation = env.reset()

        action = learner.choose_action(str(observation))

        steps = 1

        action_list = [action]

        learner.eligibility_trace *= 0

        while True:
            # fresh env
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = learner.choose_action(str(observation_))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            learner.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_

            action = action_

            action_list.append(action)

            steps += 1

            # break while loop when end of this episode
            if done:
                break

        print('epoch %d - reward %d(steps=%d)' % (episode, reward, steps))

        if steps <= 4:
            print(action_list)
        # if episode % 5 == 0:
        #     learner.export_weights('maze_epoch%d.csv' % episode)
        # print(RL.q_table)

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    # RL = QuantLearner(actions=list(range(env.n_actions)))
    # env.after(100, partial(update, RL))
    # env.mainloop()

    RL = SarsaLearner(actions=list(range(env.n_actions)))
    # RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, partial(sarsa_update, RL))
    env.mainloop()

    # RL = SarsaLambdaLeaner(actions=list(range(env.n_actions)))
    # env.after(100, partial(sarsa_lam_update, RL))
    # env.mainloop()
