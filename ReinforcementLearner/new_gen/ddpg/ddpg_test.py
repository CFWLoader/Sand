from ddpg import DDPGCritic
from torch import tensor, optim

ddpgc = DDPGCritic(5, 2, 2)

optim_int = optim.Adam(ddpgc.parameters(), lr=0.1)

# print(ddpgc.ws)
#
# print(ddpgc.wa)
#
# print(ddpgc.b1)

net_out = ddpgc.forward([2, 2, 2, 2, 2], [3, 3])

real_out = tensor([15, 15]).cuda()

td_err = real_out - net_out

loss_val = td_err.square()

print(net_out)

print(td_err)

print(loss_val)

optim_int.zero_grad()

loss_val.backward()

optim_int.step()

print(ddpgc.ws)

print(ddpgc.wa)

print(ddpgc.b1)
