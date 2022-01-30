from .mlpv import MLPV
from .mlppi import MLPPi
import torch.nn as nn

class MLPA2C(nn.Module):
    def __init__(self, state_size, n_actions, hidden_archs, hidden_acts, device):
        super().__init__()
        self.pi = MLPPi(state_size, n_actions, hidden_archs["pi"], hidden_acts["pi"], device)
        self.v = MLPV(state_size, n_actions, hidden_archs["v"], hidden_acts["v"], device)
        self.device = device
        self.to(self.device)


    def forward(self, obs, act=None):
        v = self.v(obs)
        log_p = None
        if act is None:
            act, (log_p, entropy) = self.pi.predict(obs, deterministic=False)
        else:
            log_p, entropy = self.pi.evaluate_action(obs, act)
        return act, log_p, v, entropy