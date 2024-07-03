"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from new_gen.policy_gradient.PolicyGradient import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = True  # rendering wastes time

env = gym.make('MountainCar-v0', render_mode='human')
# env = gym.make('MountainCar-v0', render_mode='human')
env.reset(seed=144003)  # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

RL.load('pg_net2k.pkl')

observation = env.reset()[0]

while True:
    if RENDER: env.render()

    action = RL.choose_action(observation)

    observation_, reward, done, info, extra_map = env.step(action)  # reward = -1 in all cases

    RL.store_transition(observation, action, reward)

    observation = observation_

    if done:
        break

env.close()