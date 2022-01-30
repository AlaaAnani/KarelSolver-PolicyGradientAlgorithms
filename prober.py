from stable_baselines3.common import policies
import torch
import numpy as np
class prober:
    def __init__(self, agent, env) -> None:
        self.agent = agent
        self.env = env
        self.act_dim = env.action_space.n
            
    def predict(self, s):
        if self.env.state_space == "S0":
            if (s[0:36] == s[36:72]).all():
                return 5, 1 # finish
        if isinstance(self.agent, policies.ActorCriticPolicy):
            s = torch.FloatTensor(np.expand_dims(s,axis=0)).to("cuda:0")
            latent_pi, _, latent_sde = self.agent._get_latent(s)
            dist = self.agent._get_action_dist_from_latent(latent_pi,latent_sde).distribution
        else:
            dist = self.agent.forward(s)
        _, a = dist.probs.topk(2)
        a = a.squeeze(0).cpu().numpy()
        _, r, _, _ = self.env.probe(s, a[0])
        if r!=-1:
            return a[0], 1
        else:
            return a[1], 0