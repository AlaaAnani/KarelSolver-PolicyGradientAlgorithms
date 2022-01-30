import itertools
from copy import deepcopy
import torch
import torch.optim as optim
from .policy_gradient_base import PolicyGradientBase


class SAC(PolicyGradientBase):
    def __init__(
        self,
        env,
        network=None,
        gamma=0.99,
        learning_rate=3e-4,
        alpha=0.5,
        max_episodes=None,
        update_every_n=1,
        epochs = 5,
        show_every_n=1e14,
        evaluate_every_n=200,
        max_eps_len=100,
        num_actions=6,
        learn_by_demo=True,
        load_pretrained=False,
        model_name = "sac",
        keep_overall_stats=True
    ):
        super().__init__(
            network=network,     
            env=env,
            gamma=gamma,
            lr=learning_rate,
            evaluate_every_n=evaluate_every_n,
            show_every_n=show_every_n,
            alpha=alpha,
            epochs = epochs,
            model_name=model_name,
            update_every_n =update_every_n,
            max_episodes=max_episodes,
            max_eps_len=max_eps_len,
            num_actions=num_actions,
            learn_by_demo=learn_by_demo,
            load_model=load_pretrained,
            keep_overall_stats=keep_overall_stats
        )
        self.target_network = deepcopy(network)
        self.target_network.eval()
        
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(network.q1.parameters(), network.q2.parameters())
        self.Poptim = optim.Adam(network.pi.parameters(), lr=learning_rate)
        self.Qoptim = optim.Adam(self.q_params, lr=learning_rate)      

    def ComputePLoss(self, replay_buffer):
        s, a =replay_buffer.states, replay_buffer.actions
        q1_pi = self.network.q1(s,a)
        q2_pi = self.network.q2(s,a)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (-self.alpha * replay_buffer.logps - q_pi).mean()
        return loss_pi

    def ComputeQLoss(self, replay_buffer):
        s, a, r, s_next,  d = replay_buffer.states, replay_buffer.actions, replay_buffer.rewards, replay_buffer.states_next, replay_buffer.donez
        # Current Qs
        q1 = self.network.q1(s,a)
        q2 = self.network.q2(s,a)
        with torch.no_grad():
            a2, (logp_a2, _) = self.network.pi.predict(s_next, deterministic=True)
            t_q1 = self.target_network.q1(s_next, a2)
            t_q2 = self.target_network.q2(s_next, a2)
            q_pi_targ = torch.min(t_q1, t_q2)
            bellman = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - bellman)**2).mean()
        loss_q2 = ((q2 - bellman)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        return loss_q

    def update_agent(self):
        Qloss = self.ComputeQLoss(self.replay_buffer)

        self.Qoptim.zero_grad()
        Qloss.backward()
        self.Qoptim.step()

        for p in self.q_params:
            p.requires_grad = False
        
        Ploss = self.ComputePLoss(self.replay_buffer)

        self.Poptim.zero_grad()
        Ploss.backward()
        self.Poptim.step()

        for p in self.q_params:
            p.requires_grad = True
        polyak=0.995
        with torch.no_grad():
            for p, p_targ in zip(self.network.parameters(), self.target_network.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        return {'pi_loss':Ploss.item(), "q_loss":Qloss.item()}

