import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import FC

class MLPV(nn.Module):
    def __init__(self,state_size, n_actions, hidden_arch, hidden_act, device):
        super().__init__()
        self.n_actions = n_actions
        sizes = [state_size] + hidden_arch + [1]
        self.v = FC(sizes, hidden_act, output_activation=nn.Identity)
        self.device = device
        self.to(device)

    def forward(self, obs):
        obs = Variable(torch.from_numpy(obs).float()).to(self.device)
        # obs = obs.view(obs.shape[0], -1)
        v = self.v(obs)
        # v = torch.squeeze(v, -1)
        return v