import torch
import torch.optim as optim
from .policy_gradient_base import PolicyGradientBase

class A2C(PolicyGradientBase):
    def __init__(
        self,
        env,
        network=None,
        gamma=0.99,
        lr=3e-4,
        alpha=0.5,
        clip_range=None,
        max_episodes=None,
        max_eps_len=100,
        num_actions=6,
        learn_by_demo=True,
        epochs = 5,
        vf_coeff = 0.5,
        ent_coeff = 0.01,
        evaluate_every_n=500,
        show_every_n=1e14,
        update_every_n =5,
        load_pretrained=False,
        model_name = "A2C",
        keep_overall_stats=True

    ):
        super().__init__(
            network=network,     
            env=env,
            gamma=gamma,
            lr=lr,
            model_name=model_name,
            evaluate_every_n=evaluate_every_n,
            show_every_n=show_every_n,
            alpha=alpha,
            epochs = epochs,
            update_every_n =update_every_n,
            max_episodes=max_episodes,
            max_eps_len=max_eps_len,
            num_actions=num_actions,
            learn_by_demo=learn_by_demo,
            load_model=load_pretrained,
            keep_overall_stats=keep_overall_stats

        )
        self.clip_range = clip_range
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.Poptim = optim.Adam(self.network.pi.parameters(), lr=self.lr)
        self.Voptim = optim.Adam(self.network.v.parameters(), lr=self.lr)



    def ComputePLoss(self, replay_buffer):
        log_pi, G, v = replay_buffer.logps, replay_buffer.Gs, replay_buffer.vs
        advantage = G - v.detach() if not self.learn_by_demo else 1
        if self.clip_range:
            c_lo, c_hi = self.clip_range
            pi_comp = torch.min(log_pi, log_pi.clamp(min=c_lo, max=c_hi))
            loss_pi = self.alpha * (-pi_comp*advantage).mean()
        else:
            loss_pi = self.alpha * (-log_pi*advantage).mean()
        return loss_pi

    def ComputeVLoss(self, replay_buffer):
        G, v = replay_buffer.Gs, replay_buffer.vs
        loss_v= self.alpha*((G - v).pow(2)).mean()
        return loss_v

    def update_agent(self):
        PLoss = self.ComputePLoss(self.replay_buffer)
        VLoss = self.ComputeVLoss(self.replay_buffer)
        ELoss = -torch.mean(self.replay_buffer.entropies)
        PLoss+=self.ent_coeff*ELoss
        # loss = PLoss+self.vf_coeff*VLoss+self.ent_coeff*ELoss
        self.Poptim.zero_grad()
        PLoss.backward()
        self.Poptim.step()

        self.Voptim.zero_grad()
        VLoss.backward()
        self.Voptim.step()

        return {"pi_loss":PLoss.item(), "v_loss": VLoss.item()}