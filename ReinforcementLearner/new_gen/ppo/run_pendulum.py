import numpy as np
import matplotlib.pyplot as plt
import gym

from new_gen.ppo.ppo import ProximalPolicyOptimization, PenaltyMethodName, DivergenceSmoothMethodConfig

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
# A_LR = 0.0001
# C_LR = 0.0002
BATCH = 32
# A_UPDATE_STEPS = 10
# C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1

env = gym.make('Pendulum-v1').unwrapped
ppo = ProximalPolicyOptimization(A_DIM, S_DIM, )
all_ep_r = []

for ep in range(EP_MAX):
    s, _ = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % ppo.dsmc.lam) if ppo.dsmc.name == PenaltyMethodName.KL_PENALTY else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
