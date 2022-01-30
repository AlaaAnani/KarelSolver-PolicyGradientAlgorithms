import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from utils import FC
import torch.nn.functional as F


class MLPPi(nn.Module):
    def __init__(self, state_size, n_actions, hidden_arch, hidden_act, device):
        super().__init__()
        sizes = [state_size] + hidden_arch + [n_actions]
        self.pi = FC(sizes, hidden_act, output_activation=nn.Softmax)
        self.n_actions = n_actions
        self.device = device
        self.to(device)

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
            obs = Variable(obs).to(self.device)
            # obs = obs.view(obs.shape[0], -1)
        logits = self.pi(obs)
        return Categorical(logits)

    def evaluate_action(self, obs, act):
        dist = self.forward(obs)
        if not isinstance(act, torch.Tensor):
            act = Variable(torch.from_numpy(act)).to(self.device)
        # act = F.one_hot(act.long(), self.n_actions).float()
        return dist.log_prob(act), dist.entropy()

    def predict(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            _, act = dist.probs.max(dim=1)
        else:
            act = dist.sample()
        return act, (dist.log_prob(act), dist.entropy())

