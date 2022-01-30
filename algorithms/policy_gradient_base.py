
from abc import abstractmethod
import torch
from utils import *
from .replay_buffer import ReplayBuffer
import pickle
class PolicyGradientBase:
    def __init__(
        self, 
        env,
        network=None,
        gamma=0.99,
        lr=3e-4,
        evaluate_every_n=500,
        show_every_n=100,
        alpha=0.5,
        update_every_n =5,
        max_episodes=None,
        max_eps_len=100,
        epochs = 5,
        num_actions=6,
        learn_by_demo=True,
        load_model=False,
        model_name=None,
        keep_overall_stats=True
    ):
        self.model_name = model_name
        self.network = network
        if load_model:
            self.load_algo_policy()
        self.evaluate_every_n = evaluate_every_n
        self.show_every_n = show_every_n
        self.env = env
        self.state_type = self.env.state_space
        self.gamma = gamma
        self.lr = lr
        self.alpha=alpha
        self.max_episodes = max_episodes
        self.max_eps_len = max_eps_len
        self.num_actions = num_actions
        self.learn_by_demo = learn_by_demo
        self.epochs = epochs
        self.replay_buffer = ReplayBuffer(update_every_n, gamma, network.device)
        self.keep_overall_stats = keep_overall_stats
        os.makedirs('models', exist_ok=True)
        if self.keep_overall_stats:
            os.makedirs('stats', exist_ok=True)
            self.overall_stats = {'pi_loss': [], 
                                'q_loss': [],
                                'v_loss': [], 
                                'solved_percentage':[],
                                'solved_optimally_percentage': [],
                                'avg_extra_steps': [],
                                'disc_returns': [],
                                'avg_reward': [],
                                'avg_disc_returns': []
                                }


    def train(self, dataset_dict, dataset_all, expert_trajectories=None):
        if self.max_episodes is None:
            self.max_episodes = len(dataset_all['train_task'])
        best_solved = 0
        best_extra_steps = 1e9
        for epoch in range(self.epochs):
            for episode in range(self.max_episodes):
                obs = self.env.reset()
                optimal_actions = expert_trajectories[self.env.current_ep]['act']
                self.network.train()
                update = self.gen_episode(obs, optimal_actions)
                if update:
                    stats = self.update_agent()
                    self.replay_buffer.reset()
                    if self.keep_overall_stats:
                        for key in stats:
                            self.overall_stats[key].append(stats[key])
                            self.overall_stats[key].append(stats[key])
                if ((episode+1) % self.evaluate_every_n) == 0:
                    eval_stats = eval_model(self.network.pi, dataset_dict['data'], 
                                show_every_n = self.show_every_n, 
                                state_type=self.state_type, 
                                H=self.max_eps_len, 
                                max_eps=None, 
                                type_='val')
                    for key in eval_stats:
                        self.overall_stats[key].append(eval_stats[key])
                    self.save_overall_stats()
                    # print(self.overall_stats)
                    print(f"E[{epoch+1}/{self.epochs}],", end="\t")
                    print(f'EP[{episode+1}/{self.max_episodes}]:', end="\t")
                    print(f"tried {len(dataset_dict['data']['val_task'])}, s={eval_stats['solved_percentage']}", end="\t")
                    print(f"os={eval_stats['solved_optimally_percentage']}", end="\t")

                    for key in stats:
                        print(f"{key}: {round(stats[key], 5)}", end="\t")
                    print()

                    if eval_stats['solved_percentage']>best_solved or eval_stats['solved_percentage']==best_solved and eval_stats['avg_extra_steps']<best_extra_steps:
                        best_solved = eval_stats['solved_percentage']
                        best_extra_steps = eval_stats['avg_extra_steps']
                        self.save_algo_policy()
        # self.load_algo_policy()
        return self.network
    def eval_model(self, ds):
        eval_stats = eval_model(self.network.pi, ds, 
            show_every_n = self.show_every_n, 
            state_type=self.state_type, 
            H=self.max_eps_len, 
            max_eps=None, 
            type_='val')
        return eval_stats
    def load_overall_stats(self):
        self.overall_stats = pickle.load(open(f'stats/{self.model_name}.pkl', 'rb'))
        return self.overall_stats
    def save_overall_stats(self):
        pickle.dump(self.overall_stats, open(f'stats/{self.model_name}.pkl', 'wb'))

    def save_algo_policy(self):
        save_path = f"models/{self.model_name}.pth"
        torch.save(self.network.state_dict(), save_path)

    def load_algo_policy(self):
        load_path = f"models/{self.model_name}.pth"
        self.network.load_state_dict(torch.load(load_path))

    def gen_episode(self, obs, optimal_actions):
        update = False
        for step in range(self.max_eps_len):
            if self.learn_by_demo:
                a = optimal_actions[step]
                a = np.array(a)
                # obs = np.expand_dims(obs, axis=0)
                a, log_p, v, e = self.network(obs, a)
            else:
                a, log_p, v, e =self.network(obs)
                a = a.cpu().detach().numpy()

            obs_next, r, d, _ = self.env.step(a)
            update |= self.replay_buffer.insert(obs, v, a, r, obs_next, log_p, d, e)
            obs = obs_next
            if d:
                break
        return update

    @abstractmethod
    def update_agent(self):
        raise NotImplementedError()
