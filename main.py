#%%
from networks.mlpsac import MLPSAC
from networks.mlpa2c import MLPA2C
from algorithms.sac import SAC
from algorithms.a2c import A2C
from Karel import *
from utils import *
import torch.nn as nn
import torch
from pickle import dump
from behavior_cloning import BC, BC_A2C


state_type = 'S0'
dataset_dict, dataset_all = load_datasets(dataset_dir="dataset")
expert_trajectories = dataset_to_trajs(dataset_dict, save_path="trajectories", traj_obj=False, state_space=state_type)
env = Karel(dataset_all['train_task'], state_type, sequential_loading=False)
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## HYPERPARAMETERS
epochs = 4
update_every_n=20
max_episodes=None
show_every_n=1e14
evaluate_every_n=2000
max_eps_len=100
num_actions=6
# BC params
batch_size = 32
epochs_bc = 15
arch_bc =[128, 128, 128]
rounds_bc = 40
steps_per_round = 10000
epochs_per_round_bc = 1


hidden_archs_sac = {
    "pi": [512, 256],
    "q" : [512, 256]
}
hidden_acts_sac = {
    "pi":nn.ReLU,
    "q": nn.ReLU
}

hidden_archs_a2c = {
    "pi": [512, 256],
    "v" : [512, 256]
}
hidden_acts_a2c = {
    "pi":nn.ReLU,
    "v": nn.ReLU
}
archs = {
    'SAC with Demo':{
    "pi": [256, 256],
    "q" : [256, 256]
}, 
    'SAC':{
    "pi": [256, 256],
    "q" : [256, 256]
}, 
    'SAC with Demo (512, 256)':{
    "pi": [512, 256],
    "q" : [512, 256]
}, 
    'SAC without Demo':{
    "pi": [128, 128, 128],
    "q" : [128, 128, 128]
}, 
    'A2C with Demo':{
    "pi": [256, 256],
    "v" : [256, 256]
}, 
'A2C':{
    "pi": [256, 256],
    "v" : [256, 256]
},
    'A2C without Demo':{
    "pi": [128, 128, 128],
    "v" : [128, 128, 128]
},   
    'A2C with Demo (512, 256)':{
    "pi": [512, 256],
    "v" : [512, 256]
}
}
# %%
#experiments = ['BC FeedForwardPolicy', 'BC A2C']#, 'SAC with Demo (512, 256)', 'A2C with Demo (512, 256)']#, 'A2C without Demo', 'SAC without Demo']
#experiments = ['SAC with Demo', 'SAC with Demo (512, 256)', 'A2C with Demo',  'A2C with Demo (512, 256)']#, 'A2C without Demo', 'SAC without Demo']
#experiments = [f'A2C with Demo (512, 256)_{epochs}']
#experiments = ['SAC without Demo', 'A2C without Demo']
experiments = ['SAC with Demo (512, 256)',  'A2C with Demo (512, 256)',"SAC", "A2C" ]
global_stats_path = 'stats/global_stats.pkl'
best_stats_path = 'stats/best_stats.pkl'
os.makedirs("stats", exist_ok=True)
global_stats, best_stats = pre_load_stats(experiments, global_stats_path, best_stats_path)
for experiment in tqdm(experiments):
    if global_stats[experiment] != None and best_stats[experiment] != None:
        continue

    learn_by_demo = False
    if 'with Demo' in experiment:
        learn_by_demo = True

    if 'SAC' in experiment:
        network = MLPSAC(state_size, n_actions, archs[experiment], hidden_acts_sac, device)
        policy = SAC(env,
                network=network,
                update_every_n=update_every_n,
                epochs = epochs,
                max_episodes=max_episodes,
                evaluate_every_n=evaluate_every_n,
                max_eps_len=max_eps_len,
                num_actions=num_actions,
                learn_by_demo=learn_by_demo,
                model_name=experiment
                )
        policy.train(dataset_dict, dataset_all, expert_trajectories)
        policy.load_algo_policy()
        best_stats_per_model = policy.eval_model(dataset_dict['data'])
        global_stats[experiment] = policy.overall_stats 
    elif 'BC FeedForwardPolicy' in experiment:
        _, _, best_stats_per_model = BC(dataset_dict, dataset_all,
                                epochs=epochs_bc, batch_size=batch_size,
                                arch=arch_bc, 
                                state_type=state_type, 
                                show_every_n = show_every_n)
    elif 'BC A2C' in experiment:
        _, _, best_stats_per_model = BC_A2C(dataset_dict, dataset_all, rounds=rounds_bc,
                        steps_per_round=steps_per_round, epochs_per_round=epochs_per_round_bc, 
                        batch_size=batch_size, arch=arch_bc, state_type=state_type)
    else:
        network = MLPA2C(state_size, n_actions, archs[experiment], hidden_acts_a2c, device)
        policy = A2C(env=env,
                network=network,
                update_every_n=update_every_n,
                epochs = epochs,
                max_episodes=max_episodes,
                evaluate_every_n=evaluate_every_n,
                max_eps_len=max_eps_len,
                num_actions=num_actions,
                learn_by_demo=learn_by_demo,
                model_name=experiment
                )

        policy.train(dataset_dict, dataset_all, expert_trajectories)
        policy.load_algo_policy()
        best_stats_per_model = policy.eval_model(dataset_dict['data'])
        global_stats[experiment] = policy.overall_stats 
    best_stats[experiment] = best_stats_per_model
    dump(global_stats, open(global_stats_path, 'wb'))
    dump(best_stats, open(best_stats_path, 'wb'))

graph_global_stats(global_stats,
                   update_every_n=update_every_n,
                   evaluate_every_n=evaluate_every_n)
graph_best_stats(best_stats)

# %%
