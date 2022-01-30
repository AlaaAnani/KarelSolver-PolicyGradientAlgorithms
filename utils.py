from imitation.data import types
from Karel import *
import numpy as np
import os
from tqdm import tqdm
import time
import json
import pickle
import plotly.graph_objects as go
import torch.nn as nn

from prober import prober

dataset_types = ['data_easy', 'data_medium', 'data']
dataset_dir = "/home/alaa/Academics/KarelTaskSolver-RL/dataset/"

def pre_load_stats(experiments, global_stats_path, best_stats_path):
    if os.path.exists(global_stats_path):
        global_stats = pickle.load(open(global_stats_path, 'rb'))
    else:
        global_stats = {}
    if os.path.exists(best_stats_path):
        best_stats = pickle.load(open(best_stats_path, 'rb'))
    else:
        best_stats = {}
    for exp in experiments:
        global_stats[exp] = None
        best_stats[exp] = None
    # pre_load stats if already there


    pre_computed_stats = [file.split('.')[0] for file in os.listdir('stats') if 'global' not in file and 'best' not in file]
    if len(pre_computed_stats) != 0:
        for experiment in pre_computed_stats:
            global_stats[experiment] = pickle.load(open(f'stats/{experiment}.pkl', 'rb'))
    return global_stats, best_stats

def FC(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        layers.append(activation() if j < len(sizes) - 2 else output_activation(dim=-1))
    return nn.Sequential(*layers)

def load_datasets(dataset_dir="/home/alaa/Academics/KarelTaskSolver-RL/dataset/",
                  types=['data_easy', 'data_medium', 'data']):
    dataset_dict = {}
    sub_dict = {}
    for type_ in types:
        for tr_val in ['train', 'val']:
            base_path = os.path.join(dataset_dir, type_, tr_val)
            files = [os.path.join(base_path, 'task', file) for file in
                     sorted(os.listdir(os.path.join(base_path, 'task')))]
            sub_dict[f'{tr_val}_task'] = files
            files = [os.path.join(base_path, 'seq', file) for file in
                     sorted(os.listdir(os.path.join(base_path, 'seq')))]
            sub_dict[f'{tr_val}_seq'] = files
        dataset_dict[type_] = sub_dict
        sub_dict = {}
    all_tr_task_pths = []
    all_val_task_pths = []
    all_tr_seq_paths = []
    all_val_seq_paths = []

    for type_ in dataset_dict:
        all_tr_task_pths.extend(dataset_dict[type_]['train_task'])
        all_val_task_pths.extend(dataset_dict[type_]['val_task'])
        all_tr_seq_paths.extend(dataset_dict[type_]['train_seq'])
        all_val_seq_paths.extend(dataset_dict[type_]['val_seq'])

    dataset_all = {'train_task': all_tr_task_pths,
                   'val_task': all_val_task_pths,
                   'train_seq': all_tr_seq_paths,
                   'val_seq': all_val_seq_paths}

    return dataset_dict, dataset_all


def dataset_to_npz(dataset_dict, save_path="/home/alaa/Academics/KarelTaskSolver-RL/dataset_npz/", state_space='S1'):
    if all([os.path.exists(os.path.join(save_path, f'{x}_{state_space}.npz')) for x in
            ['data_easy', 'data_medium', 'data', 'all']]):
        return
    os.makedirs(save_path, exist_ok=True)
    numpy_dict_all = {}
    for data_type in tqdm(dataset_dict):
        train_tasks = dataset_dict[data_type]['train_task']
        train_seqs = dataset_dict[data_type]['train_seq']

        rewards = []
        episode_starts = []
        reward_sum = []
        observations = []
        episode_returns = np.zeros((len(train_tasks),))
        numpy_actions = []

        ep_idx = 0
        episode_starts.append(True)
        reward_sum = 0.0

        for task_pth, seq_pth in zip(train_tasks, train_seqs):
            with open(seq_pth, 'r') as f:
                data = json.load(f)
            actions_str = data['sequence']
            actions_ls = [actions_dict[action_str] for action_str in actions_str]

            env = Karel([task_pth], state_space)
            obs = env.reset()

            observations.append(obs)
            for action in actions_ls:
                numpy_actions.append(action)
                obs, reward, done, _ = env.step(action)
                # env.render()

                rewards.append(reward)
                episode_starts.append(done)
                reward_sum += reward
                if done:
                    env.close()
                    episode_returns[ep_idx] = reward_sum
                    reward_sum = 0.0
                    ep_idx += 1
                else:
                    observations.append(obs)
        numpy_actions = np.array(numpy_actions).reshape((-1, 1))
        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts[:-1])
        observations = np.array(observations)
        numpy_dict = {
            'actions': numpy_actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }
        if len(numpy_dict_all) == 0:
            numpy_dict_all = numpy_dict
        else:
            for key in numpy_dict:
                numpy_dict_all[key] = np.concatenate((numpy_dict_all[key], numpy_dict[key]))
        if save_path is not None:
            full_path = os.path.join(save_path, f'{data_type}_{state_space}.npz')
            np.savez(full_path, **numpy_dict)
            print("Saved ", full_path)
    if save_path is not None:
        full_path = os.path.join(save_path, f'all_{state_space}.npz')
        np.savez(full_path, **numpy_dict_all)
        print("Saved ", full_path)


def dataset_to_trajs(dataset_dict, save_path, traj_obj=True, state_space='S1'):
    trajstr = "trobj" if traj_obj else 'traj'
    if all([os.path.exists(os.path.join(save_path, f'{x}_{state_space}_{trajstr}.pkl')) for x in
            ['data_easy', 'data_medium', 'data', 'all']]):
        with open(os.path.join(save_path, f'all_{state_space}_{trajstr}.pkl'), "rb") as f:
            all_dataset = pickle.load(f)
        return all_dataset
    os.makedirs(save_path, exist_ok=True)
    all_dataset = []
    for data_type in tqdm(dataset_dict):
        dataset = []
        train_tasks = dataset_dict[data_type]['train_task']
        train_seqs = dataset_dict[data_type]['train_seq']

        for task_pth, seq_pth in zip(train_tasks, train_seqs):
            rewards = []
            observations = []
            actions = []
            with open(seq_pth, 'r') as f:
                data = json.load(f)
            actions_str = data['sequence']
            actions_ls = [actions_dict[action_str] for action_str in actions_str]

            env = Karel([task_pth], state_space)
            obs = env.reset()

            observations.append(obs)
            for action in actions_ls:
                actions.append(action)
                obs, reward, done, _ = env.step(action)
                rewards.append(float(reward))
                if done:
                    env.close()
                observations.append(obs)
            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            if traj_obj:
                traj = types.TrajectoryWithRew(obs=observations, acts=actions, rews=rewards, infos=None)
                # traj = types.Trajectory(obs=observations, acts=actions, infos=None)
            else:
                traj = {"obs": observations[:-1],"act":actions, "rews":rewards }

            dataset.append(traj)
        if save_path is not None:
            full_path = os.path.join(save_path, f'{data_type}_{state_space}_{trajstr}.pkl')
            pickle.dump(dataset, open(full_path, "wb+"))
            print("Saved ", full_path)
        all_dataset.extend(dataset)
    if save_path is not None:
        full_path = os.path.join(save_path, f'all_{state_space}_{trajstr}.pkl')
        pickle.dump(all_dataset, open(full_path, "wb+"))
        print("Saved ", full_path)
    return all_dataset


def dataset_to_dict(dataset_dict, save_path="/home/alaa/Academics/KarelTaskSolver-RL/dataset_pkl/", state_space='S1'):
    if all([os.path.exists(os.path.join(save_path, f'{x}_{state_space}_dict.pkl')) for x in
            ['data_easy', 'data_medium', 'data', 'all']]):
        return
    os.makedirs(save_path, exist_ok=True)
    all_dataset = {}
    for data_type in tqdm(dataset_dict):
        dataset = {}
        train_tasks = dataset_dict[data_type]['train_task']
        train_seqs = dataset_dict[data_type]['train_seq']

        for task_pth, seq_pth in zip(train_tasks, train_seqs):

            with open(seq_pth, 'r') as f:
                data = json.load(f)
            actions_str = data['sequence']
            actions_ls = [actions_dict[action_str] for action_str in actions_str]

            env = Karel([task_pth], state_space)
            obs = env.reset()

            for action in actions_ls:
                dataset[str(obs)] = action
                obs, reward, done, _ = env.step(action)
                if done:
                    env.close()

        if save_path is not None:
            full_path = os.path.join(save_path, f'{data_type}_{state_space}_dict.pkl')
            pickle.dump(dataset, open(full_path, "wb+"))
            print("Saved ", full_path)
        all_dataset = {**all_dataset, **dataset}
    if save_path is not None:
        full_path = os.path.join(save_path, f'all_{state_space}_dict.pkl')
        pickle.dump(all_dataset, open(full_path, "wb+"))
        print("Saved ", full_path)


def eval_model(model, dataset_dict, 
    show_every_n = 500, 
    state_type='S1', 
    H=100, 
    max_eps=None, 
    type_='val',
    gamma=0.99):
    # eval params 
    total_solved = 0
    total_optimally_solved = 0
    total_extra_steps = 0
    episodes_stats = {}
    overall_stats = {}


    avg_reward = []
    
    if max_eps is None:
    # make env
        env = Karel(dataset_dict[f'{type_}_task'], state_space=state_type, sequential_loading=True)
        n_episodes = len(dataset_dict[f'{type_}_task'])
    else:
        env = Karel(dataset_dict[f'{type_}_task'][:max_eps], state_space=state_type, sequential_loading=True)
        n_episodes = len(dataset_dict[f'{type_}_task'][:max_eps])      
    returns_ls = []
    for ep in range(n_episodes):
        optimal_seq = json.load(open(dataset_dict[f'{type_}_seq'][ep], 'r'))['sequence']
        optimal_actions_len = len(optimal_seq)
        obs = env.reset()
        done = False

        steps = 0 # max number of steps before timeout 
        action_ls = []
        reward_ls = []


        optimally_solved = False
        solved = False # false if crashed or timeout
        G = 0
        disc = 1
        agent = prober(model, env)
        while not done and steps < H:
            action, _ = agent.predict(obs)
            obs, rewards, done, info = env.step(action)
            G  +=disc*rewards
            disc*=gamma
            if (ep+1) % show_every_n == 0:
                time.sleep(1)
                #clear_output(wait=True)
                env.render()

            # Keep stats
            action_ls.append(int(action))
            reward_ls.append(int(rewards))

            if done: # check if finish or crash
                solved = env.solved
                total_solved+=1 if solved else 0
                optimally_solved = env.solved and len(action_ls) == optimal_actions_len
                total_optimally_solved += 1 if optimally_solved else 0
                total_extra_steps += len(action_ls) - optimal_actions_len if solved else 0

            else:
                steps +=1

        episodes_stats[ep] = {  'action_ls':action_ls,
                                'reward_ls':reward_ls,
                                'solved':solved,
                                'optimally_solved':optimally_solved,
                                'cum_reward': sum(reward_ls),
                                'avg_reward': sum(reward_ls)/(len(reward_ls)+1e-13),
                                'disc_return': G}
        returns_ls.append(G)

        avg_reward.append(sum(reward_ls)/(len(reward_ls)+1e-13))

    overall_stats['avg_reward'] = sum(avg_reward)/n_episodes
    overall_stats['avg_extra_steps'] = float(total_extra_steps)/float(total_solved+1e-13)
    overall_stats['solved_percentage'] = total_solved/n_episodes
    overall_stats['solved_optimally_percentage'] = total_optimally_solved/n_episodes
    overall_stats['avg_disc_returns'] = sum(returns_ls)/n_episodes
    return overall_stats

def graph_line(x, y, title, xaxis_title, yaxis_title, traces_names):
    os.makedirs('graphs', exist_ok=True)
    fig = go.Figure()
    for i, trace_name in enumerate(traces_names):
        fig.add_trace(go.Scatter(x=x, y=y[i],
                                 name=trace_name))
    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_x=0.5)
    fig.write_image(f"graphs/{title}.png")

def graph_bar(x, y, title, xaxis_title, yaxis_title):
    os.makedirs('graphs', exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        text=[str("{:.4f}".format(a)) for a in y]
    ))
    if 'steps' in title:
        xy1 = min(y)
    else:
        xy1 = max(y)
    fig.add_shape(type="line",
                x0=-1, y0=xy1, x1=len(y),
                y1=xy1,
                line=dict(color='black', width=2, dash="dashdot")
                )
    fig.update_layout(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_x=0.5)
    fig.write_image(f"graphs/{title}.png")




def graph_global_stats(global_stats=None, global_stats_path=None, update_every_n=None, evaluate_every_n=None):
    if global_stats is None and os.path.exists(global_stats_path):
        global_stats = pickle.load(open(global_stats_path, 'rb'))
    model_configs = list(global_stats.keys())
    sac_models = [config for config in model_configs if 'SAC' in config]
    a2c_models = [config for config in model_configs if 'A2C' in config]
    sac_a2c_models = sac_models + a2c_models
    # SAC
    losses_keys = ['q_loss', 'pi_loss']
    losses = []
    for loss in losses_keys:
        losses = []
        for model_config in sac_models:
            losses.append(global_stats[model_config][loss])
        graph_line(x=np.arange(len(losses[0])),
                    y=losses,
                    title=f'SAC {loss} Vs. Update Steps Across Different Configs',
                    xaxis_title=f'Update Step Number (every {update_every_n} episodes)',
                    yaxis_title=f'{loss}',
                    traces_names=sac_models)
    # A2C
    losses_keys = ['v_loss', 'pi_loss']
    losses = []
    for loss in losses_keys:
        losses = []
        for model_config in a2c_models:
            losses.append(global_stats[model_config][loss])
        graph_line(x=np.arange(len(losses[0])),
                    y=losses,
                    title=f'A2C {loss} Vs. Update Steps  Across Different Configs',
                    xaxis_title=f'Update Step Number (every {update_every_n} episodes)',
                    yaxis_title=f'{loss}',
                    traces_names=a2c_models)
    

    ### solved and optimally evaluate_every_n for both SAC and A2C
    s_os_keys = ['solved_percentage', 'solved_optimally_percentage', 'avg_extra_steps', 'avg_disc_returns', 'avg_reward']
    for s_os in s_os_keys:
        s = []
        for model_config in sac_a2c_models:
            s.append(global_stats[model_config][s_os])
        graph_line(x=np.arange(len(s[0])),
                    y=s,
                    title=f'{s_os} Vs. Eval Steps Across Different Configs',
                    xaxis_title=f'Evaluation Step Number (every {evaluate_every_n} episodes)',
                    yaxis_title=f'{s_os}',
                    traces_names=sac_a2c_models)
    
    ### 

def graph_best_stats(best_stats=None, best_stats_path=None):
    if best_stats is None and os.path.exists(best_stats_path):
        best_stats = pickle.load(open(best_stats_path, 'rb'))
    avg_reward_per_config = []
    solved_optimally_percentage_per_config = []
    solved_percentage_per_config = []
    avg_extra_steps_per_config = []

    model_configs = list(best_stats.keys())
    ys = []
    keys = ['avg_reward', 'solved_optimally_percentage', 'solved_percentage', 'avg_extra_steps']
    for key in keys:
        ys = []
        for model_config in model_configs:
            ys.append(best_stats[model_config][key])
        graph_bar(x=model_configs,
                    y=ys,
                    title=f'{key} Vs. Models',
                    xaxis_title=f'Model Type',
                    yaxis_title=f'{key}')


        
    





def graph_eval_stats(models_stats=None, model_stats_file_path=None):
    if models_stats is None and os.path.exists(model_stats_file_path):
        models_stats = json.load(open(model_stats_file_path, 'rb'))
    # convert stats to plotable lists per model configuration
    model_configs = list(models_stats.keys())
    avg_reward_per_config = []
    solved_optimally_percentage_per_config = []
    solved_percentage_per_config = []
    avg_extra_steps_per_config = []
    train_loss_per_config = []
    val_loss_per_config = []
    for model_config in model_configs:
        avg_reward_per_config.append(models_stats[model_config]['avg_reward'])
        solved_optimally_percentage_per_config.append(models_stats[model_config]['solved_optimally_percentage'])
        solved_percentage_per_config.append(models_stats[model_config]['solved_percentage'])
        avg_extra_steps_per_config.append(models_stats[model_config]['avg_extra_steps'])

    # plot avg reward per model
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_configs,
        y=avg_reward_per_config,
        text=[str("{:.4f}".format(a)) for a in avg_reward_per_config],
        marker_color='crimson'
    ))
    fig.update_traces(textposition='inside')
    # fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(
        title_text='Average Validation Reward per Model',
        xaxis_title='Model Type',
        yaxis_title='Average Reward',
        title_x=0.5)
    fig.write_image("graphs/avg_reward.png")

    # plot optimal solved %
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_configs,
        y=solved_percentage_per_config,
        name='Solved %',
        marker_color='lightslategrey',
        text=[str("{:.4f}".format(a * 100)) for a in solved_percentage_per_config]
    ))

    fig.add_shape(type="line",
                  x0=-1, y0=max(solved_percentage_per_config), x1=len(model_configs),
                  y1=max(solved_percentage_per_config),
                  line=dict(color='black', width=2, dash="dashdot")
                  )
    fig.add_trace(go.Bar(
        x=model_configs,
        y=solved_optimally_percentage_per_config,
        name='Optimally Solved %',
        marker_color='crimson',
        text=[str("{:.4f}".format(a * 100)) for a in solved_optimally_percentage_per_config]
    ))
    fig.add_shape(type="line",
                  x0=-1, y0=max(solved_optimally_percentage_per_config), x1=len(model_configs),
                  y1=max(solved_optimally_percentage_per_config),
                  line=dict(color="LightSeaGreen", width=2, dash="dashdot")
                  )
    fig.update_layout(
        title_text='Percentage of Solved and Optimally Solved Tasks vs. Model Type',
        xaxis_title='Model Type',
        yaxis_title='Percentage of Validation Tasks',
        title_x=0.5)
    # fig.update_layout(barmode='relative', title_text='Relative Barmode')
    fig.write_image("graphs/solved_op.png")

    # plot avg extra steps per config
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_configs,
        y=avg_extra_steps_per_config,
        text=[str("{:.4f}".format(a)) for a in avg_extra_steps_per_config]
    ))

    fig.update_layout(
        title_text='Average Extra Steps per Model',
        xaxis_title='Model Type',
        yaxis_title='Average Extra Steps ',
        title_x=0.5)
    fig.write_image("graphs/avg_extra_steps.png")
    # plot train and validation losses
    # per every config, plot episodes vs loss
    fig = go.Figure()
    for i, model_config in enumerate(model_configs):
        fig.add_trace(go.Scatter(x=np.arange(len(train_loss_per_config)), y=train_loss_per_config[i],
                                 name=model_config + ' train loss'))
        fig.update_layout(
            title_text='Behavior Cloning: Training Loss per Model',
            xaxis_title='Model Type',
            yaxis_title='Training Loss',
            title_x=0.5)
        fig.write_image("graphs/train_loss.png")
    fig = go.Figure()
    for i, model_config in enumerate(model_configs):
        fig.add_trace(go.Scatter(x=np.arange(len(val_loss_per_config)), y=val_loss_per_config[i],
                                 name=model_config + ' val loss'))
        fig.update_layout(
            title_text='Behavior Cloning: Validation Loss per Model',
            xaxis_title='Model Type',
            yaxis_title='Validation Loss',
            title_x=0.5)
        fig.write_image("graphs/val_loss.png")

