import torch.nn

from ddpg import DeepDeterministicPolicyGradient
from torch import tensor, optim

ddpgc = DeepDeterministicPolicyGradient(5, 2, 2)

loss_fun = torch.nn.MSELoss()

optim_int = optim.Adam(ddpgc.target_critic.parameters(), lr=0.1)

print('===training===')
print(ddpgc.target_critic)

# net_out = ddpgc.target_critic.forward(torch.tensor([2, 2, 2, 2, 2, 3, 3], dtype=torch.float32).cuda())
# real_out = tensor([15, 15], dtype=torch.float32).cuda()
# loss_val = loss_fun(net_out, real_out)
# print(net_out)
# print(loss_val)
# optim_int.zero_grad()
# loss_val.backward()
# optim_int.step()
# print(ddpgc.target_critic.forward(torch.tensor([2, 2, 2, 2, 2, 3, 3], dtype=torch.float32).cuda()).detach())
