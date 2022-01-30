from typing import Tuple
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, num_eps,gamma, device) -> None:
        self.reset()
        self.max_eps = num_eps
        self.gamma = gamma
        self.device = device

    def insert(self, state, v, action, reward, state_next, logp, done:bool, entropy)-> bool:
        self.states.append(state)
        self.vs.append(v)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_next.append(state_next)
        self.logps.append(logp)
        self.donez.append(done)
        self.entropies.append(entropy)
        self.current += 1 if done else 0
        ready = self.current == self.max_eps
        if ready:
            self.torchify()
        return ready
            
    def reset(self):
        self.states = []
        self.vs= []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.logps = []
        self.donez = []
        self.entropies = []
        self.current = 0

    def torchify(self):
        reward_lists = self.split_rewards()
        self.states = torch.from_numpy(np.array(self.states)).float().to(self.device)
        self.vs = torch.stack(self.vs)
        self.actions = torch.from_numpy(np.array(self.actions)).float().to(self.device)
        self.rewards  = torch.from_numpy(np.array(self.rewards)).float().to(self.device)
        self.states_next = torch.from_numpy(np.array(self.states_next)).float().to(self.device)
        self.logps = torch.stack(self.logps)
        self.entropies = torch.stack(self.entropies)
        self.donez = torch.from_numpy(np.array(self.donez)).float().to(self.device)
        self.Gs = torch.cat([self.discount_rewards(reward_list)for reward_list in reward_lists])
    def split_rewards(self):
        st = 0
        en = 0
        reward_lists = []
        for i in range(len(self.rewards)):
            if self.donez[i]:
                en = i+1
                reward_lists.append(self.rewards[st:en])
                st = en
        return reward_lists
    def discount_rewards(self, rewards):
        G = 0.0
        Gs = np.zeros(len(rewards), dtype=float)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            Gs[t] = G
        return torch.FloatTensor(Gs).to(self.device)
    