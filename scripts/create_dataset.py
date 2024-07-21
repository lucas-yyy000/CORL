import numpy as np
import minari
import gymnasium as gym
from utils import *
from gymnasium.spaces import Dict, Box, Discrete
from minari.serialization import serialize_space


data_path = "/home/lucas/Workspace/Guidance for Evasion/data/"
processed_data_path = "/home/lucas/Workspace/CORL/offline_data/"
data_num = 50
map_range = 475
radar_radius = 75
img_size = 100
aircraft_detection_range = 150
grid_size=2*aircraft_detection_range/img_size
time_max = 100
time_interval = 0.5
observation_img_size = [1, img_size, img_size]

def generate_episode_data(data):
    init_state = data['start_state']
    goal_location = data['goal_location']
    radar_locs = data['radar_locations']
    radar_orientations = data['radar_orientations']
    path = data['state_history']
    inputs = data['input_history']
    risks = data['risk_history']
    # print(path.shape)
    # print(len(risks))
    observations = []
    # next_observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []
    last_idx = 0
    sars_data = []
    for i in range(len(inputs)-1):
        last_idx += 1
        heat_map = get_radar_heat_map(path[i], radar_locs, img_size, 
                                 aircraft_detection_range, grid_size, radar_radius)
        # print(heat_map.shape)
        observations.append({'heat_map': heat_map, 
                            'goal_direction': goal_location - path[i][:2],
                            'time_spent': i})
        
        actions.append(inputs[i+1])

        if i >= time_max and np.linalg.norm(goal_location - path[i][:2]) > 30.0:
            rewards.append(-risks[i] - 100)
            truncations.append(False)
            terminations.append(True)
            break
        elif i < time_max and np.linalg.norm(goal_location - path[i][:2]) <= 30.0:
            rewards.append(-risks[i] + 100)
            truncations.append(False)
            terminations.append(True)
            break
        elif i < time_max and np.linalg.norm(goal_location - path[i][:2]) > 30.0:
            rewards.append(-risks[i])
            truncations.append(False)
            terminations.append(False)
        else:
            rewards.append(-risks[i] - (i - time_max))
            truncations.append(False)
            terminations.append(True)
            break

    heat_map = get_radar_heat_map(path[last_idx], radar_locs, img_size, 
                                      aircraft_detection_range, grid_size, radar_radius)
    observations.append({'heat_map': heat_map, 
                    'goal_direction': goal_location - path[last_idx][:2],
                    'time_spent': last_idx})
        
    # print(len(observations))
    # print(len(actions))
    # print(len(rewards))
    # print(len(terminations))
    # print(len(truncations))
    for i in range(len(actions)):
        sars_data.append({'observation': observations[i],
                          'next_observation': observations[i+1],
                          'action': actions[i],
                          'reward': rewards[i],
                          'termination': terminations[i]})
        
    # episode_data = {'observations': observations,
    #                 'next_observations':  next_observations,
    #                 'actions': np.asarray(actions), 
    #                 'rewards': np.asarray(rewards), 
    #                 'terminations': np.asarray(terminations), 
    #                 'truncations': np.asarray(truncations)}

    return sars_data

import pickle 

dataset = None
u_min = np.inf
u_max = -np.inf
offline_dataset_size = 0
for i in range(data_num):
    episode_dict = np.load(data_path + f'episode_{i}.npy',allow_pickle='TRUE').item()
    sars_data = generate_episode_data(episode_dict)

    for data in sars_data:
        with open(processed_data_path + f'data_{offline_dataset_size}.pkl', 'wb') as f:
            pickle.dump(data, f)
            offline_dataset_size += 1


    # if min(episode_data['actions']) < u_min:
    #     u_min = min(episode_data['actions'])
    # if max(episode_data['actions']) > u_max:
    #     u_max = max(episode_data['actions'])
    # if not dataset:
    #     dataset =  minari.create_dataset_from_buffers("radar-dataset-v0", buffer=[episode_data], 
    #                                                 action_space=gym.spaces.Box(low=1.2*u_min, high=1.2*u_max, shape=(1,)), 
    #                                                 observation_space=Dict({"heat_map": Box(0, 255, observation_img_size), 
    #                                                                         "goal_direction": Box(-250, 250, shape=(2,)),
    #                                                                         'time_spent': Discrete(time_max + 1)})
                                                    # )
    # else:
    #     dataset.update_dataset_from_buffer([episode_data])

