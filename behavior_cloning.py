import pickle
from stable_baselines3.common import policies
from Karel import Karel
from utils import dataset_to_trajs, eval_model, load_datasets
import pickle
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

def BC(dataset_dict, dataset_all, epochs=40, batch_size=32, arch=[128, 128, 128], state_type="S0", show_every_n = int(1e9)):
    logger.configure("logs")

    dataset_to_trajs(dataset_dict, save_path="data", state_space=state_type)
    with open(f'data/all_{state_type}_trobj.pkl', "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)
    env = Karel(dataset_all['train_task'], state_type)
    env = DummyVecEnv([lambda: env])
    global FeedForwardPolicy
    class FeedForwardPolicy(policies.ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, net_arch=arch)
    bc.BC.DEFAULT_BATCH_SIZE = batch_size
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        expert_data=transitions,
        policy_class=FeedForwardPolicy
    )
    
    bc_trainer.train(n_epochs=epochs)
    stats = eval_model(bc_trainer.policy, dataset_dict['data'], show_every_n=show_every_n, state_type=state_type)
    solved_optimally_percentage = stats['solved_optimally_percentage']
    solved_percentage = stats['solved_percentage']
    print(f"solved_optimally_percentage= {solved_optimally_percentage} solved_percentage= {solved_percentage}")
    policy_path = f"models/FeedForward{str(arch)}_{epochs}Epochs_{batch_size}Batch_{solved_percentage}.zip"
    bc_trainer.save_policy(policy_path)
    return policy_path, bc_trainer.policy, stats


def BC_A2C(dataset_dict, dataset_all, rounds=40, steps_per_round=10000,epochs_per_round=1, 
            batch_size=32, arch=[128, 128, 128], state_type="S0"):
    logger.configure("logs")
    
    dataset_to_trajs(dataset_dict, save_path="data", state_space=state_type)
    with open(f'data/all_{state_type}_traj.pkl', "rb") as f:
        trajectories = pickle.load(f)
    global FeedForwardPolicy
    class FeedForwardPolicy(policies.ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, net_arch=arch)
    bc.BC.DEFAULT_BATCH_SIZE = batch_size
    env = Karel(dataset_all['train_task'], state_type)
    model = A2C(FeedForwardPolicy, env, verbose=1)
    transitions = rollout.flatten_trajectories(trajectories)
    show_every = int(1e9)
    best_solved = 0
    best_optimal = 0
    stat_list = []
    for round in range(rounds):
        model.learn(steps_per_round)
        stats = eval_model(model.policy, dataset_all, show_every_n=show_every, state_type=state_type)
        solved_optimally_percentage = stats['solved_optimally_percentage']
        solved_percentage = stats['solved_percentage']
        print(
            f"{round} - A2C -- "
            f"solved_optimally_percentage= {solved_optimally_percentage} "
            f"solved_percentage= {solved_percentage}")
        stat_list.append((solved_percentage, solved_optimally_percentage))
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            expert_data=transitions,
            policy=model.policy)
        bc_trainer.train(n_epochs=epochs_per_round)
        stats = eval_model(bc_trainer.policy, dataset_all, show_every_n=50000000, state_type=state_type)
        solved_optimally_percentage = stats['solved_optimally_percentage']
        solved_percentage = stats['solved_percentage']
        print(
            f"{round} - BC -- "
            f"solved_optimally_percentage= {solved_optimally_percentage} "
            f"solved_percentage= {solved_percentage}")
        stat_list.append((solved_percentage, solved_optimally_percentage))
        if best_solved < solved_percentage or solved_percentage==best_solved and best_optimal<solved_optimally_percentage:
                policy_path = f"BCA2C-FeedForward{str(arch)}_{epochs_per_round}Epochs_{batch_size}Batch_{solved_percentage}.zip"
                best_solved = solved_percentage
                best_optimal = solved_optimally_percentage
                bc_trainer.save_policy(policy_path)
        model.policy = bc_trainer.policy
    return model.policy, stat_list, stats
if __name__ == "__main__":
    dataset_dict, dataset_all = load_datasets(dataset_dir="dataset")
    best_model_path = "FeedForward_S0_[128, 128, 128]_15_32_0.89125_0.8791666666666667.zip"
    policy = bc.reconstruct_policy(best_model_path)
    stats = eval_model(policy, dataset_dict['data'], state_type="S0", show_every_n=1e10)
    print(stats)
