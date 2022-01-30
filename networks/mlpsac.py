from .mlpq import MLPQ
from .mlppi import MLPPi
import torch.nn as nn
import torch

class MLPSAC(nn.Module):
    def __init__(self, state_size, n_actions, hidden_archs, hidden_acts, device):
        super().__init__()
        self.pi = MLPPi(state_size, n_actions, hidden_archs["pi"], hidden_acts["pi"], device)
        self.q1= MLPQ(state_size, n_actions, hidden_archs["q"], hidden_acts["q"], device)
        self.q2= MLPQ(state_size, n_actions, hidden_archs["q"], hidden_acts["q"], device)
        self.device = device
        self.to(self.device)


    def forward(self, obs, act=None):
        log_p = None
        entropy = None
        if act is None:
            act, (log_p, entropy) = self.pi.predict(obs, deterministic=False)
        else:
            log_p, entropy = self.pi.evaluate_action(obs, act)
        # q1 = self.q1(obs, act)
        # q2 = self.q2(obs, act)
        return act, log_p, torch.tensor([0]), entropy