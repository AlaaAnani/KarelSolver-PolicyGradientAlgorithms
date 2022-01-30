
#%%
from imitation.algorithms import bc
from algorithms.sac import SAC
from algorithms.a2c import A2C
from networks.mlpsac import MLPSAC
from networks.mlpa2c import MLPA2C
from stable_baselines3.common import policies
from utils import *
import torch
dataset_dict, dataset_all = load_datasets(dataset_dir="dataset")
hidden_acts_a2c = {
    "pi":nn.ReLU,
    "v": nn.ReLU
}
archs = { 
    'SAC':{
        "pi": [256, 256],
        "q" : [256, 256]
    }, 
    'SAC with Demo (512, 256)':{
        "pi": [512, 256],
        "q" : [512, 256]
    }, 
    'A2C':{
        "pi": [256, 256],
        "v" : [256, 256]
    },   
    'A2C with Demo (512, 256)':{
        "pi": [512, 256],
        "v" : [512, 256]
    }
}
hidden_acts_sac = {
    "pi":nn.ReLU,
    "q": nn.ReLU
}
hidden_acts_a2c = {
    "pi":nn.ReLU,
    "v": nn.ReLU
}
state_type = 'S0'
dataset_dict, dataset_all = load_datasets(dataset_dir="dataset")
expert_trajectories = dataset_to_trajs(dataset_dict, save_path="trajectories", traj_obj=False, state_space=state_type)
env = Karel(dataset_all['train_task'], state_type, sequential_loading=False)
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% BC
arch = [128,128, 128]
class FeedForwardPolicy(policies.ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=arch)
best_model_path = "models/FeedForward_S0_[128, 128, 128]_15_32_0.89125_0.8791666666666667.zip"
policy = bc.reconstruct_policy(best_model_path)
stats = eval_model(policy, dataset_dict['data'], state_type="S0", show_every_n=1e10)
print(stats)
#%% SAC
for experiment in ['SAC with Demo (512, 256)']:
    network = MLPSAC(state_size, n_actions, archs[experiment], hidden_acts_sac, device)
    policy = SAC(env,network=network,model_name=experiment)
    policy.load_algo_policy()
    stats = eval_model(policy.network.pi, dataset_dict['data'], state_type="S0", show_every_n=1e10)
    print(stats)
#%% A2C
for experiment in ['A2C with Demo (512, 256)']:
    network = MLPA2C(state_size, n_actions, archs[experiment], hidden_acts_a2c, device)
    policy = A2C(env,network=network,model_name=experiment)
    policy.load_algo_policy()
    stats = eval_model(policy.network.pi, dataset_dict['data'], state_type="S0", show_every_n=1e10)
    print(stats)
# %%
