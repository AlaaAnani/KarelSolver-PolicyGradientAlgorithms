import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import FC
import torch.nn.functional as F


class MLPQ(nn.Module):
    def __init__(self,state_size, n_actions, hidden_arch, hidden_act, device):
        super().__init__()
        self.n_actions = n_actions
        sizes = [state_size+n_actions] + hidden_arch + [1]
        self.q = FC(sizes, hidden_act, output_activation=nn.Identity)
        self.device = device
        self.to(device)

    def forward(self, obs, act):
        if not isinstance(obs, torch.Tensor):
            obs = Variable(torch.from_numpy(obs))
            # obs = obs.view(obs.shape[0], -1)
        if not isinstance(act, torch.Tensor):
            act = Variable(torch.from_numpy(act))
        act = F.one_hot(act.long(), self.n_actions).float()
        q_in = torch.cat([obs, act], dim=-1).to(self.device)
        q = self.q(q_in)
        # q = torch.squeeze(q, -1)
        return q