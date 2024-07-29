import numpy as np
import minari
import gymnasium as gym
from utils import *
from gymnasium.spaces import Dict, Box, Discrete
from minari.serialization import serialize_space
import statistics
import yaml

data_config_path = "/home/lucas/Workspace/evasion_guidance/evasion_guidance/" + "params/data_collection.yaml"
with open(data_config_path,"r") as file_object:
    data_config = yaml.load(file_object,Loader=yaml.SafeLoader)


data_path = data_config['data_collection']['output_path']

map_range = data_config['env']['map_range']
radar_radius = data_config['env']['radar_radius']
min_num_radar = data_config['env']['min_num_radar']
max_num_radar = data_config['env']['max_num_radar']
V = data_config['planner']['V']
L1 = data_config['planner']['L1']
time_interval = data_config['planner']['delta_t']

with open("/home/lucas/Workspace/CORL/params/offline_data_config.yaml","r") as file_object:
    offline_data_config = yaml.load(file_object,Loader=yaml.SafeLoader)

processed_data_path = offline_data_config['output_path']
data_num = offline_data_config['load_data_num']
img_size = offline_data_config['env']['img_size']
aircraft_detection_range = offline_data_config['env']['aircraft_detection_range']
grid_size=2*aircraft_detection_range/img_size
time_max = offline_data_config['env']['max_time_step']
observation_img_size = [1, img_size, img_size]
goal_tolerance = offline_data_config['goal_tolerance']
time_scaling = offline_data_config['env']['time_scaling']

def center_goal(state, goal_location):
    state = np.array([
            [np.cos(state[2]), -np.sin(state[2]), state[0]],
            [np.sin(state[2]), np.cos(state[2]), state[1]],
            [0, 0, 1]
        ])
    state_inv = np.linalg.inv(state)
    goal_hom = np.array([goal_location[0], goal_location[1], 1])
    return np.dot(state_inv, goal_hom)[:2]

def generate_episode_data(data):
    init_state = data['start_state']
    goal_location = data['goal_location']
    radar_locs = data['radar_locations']
    radar_orientations = data['radar_orientations']
    path = data['state_history']
    inputs = data['input_history']
    risks = data['risk_history']

    #############################################################
    ###       Print some stats about the collected data       ###
    #############################################################

    print("Episode length: ", len(path))
    print("Risks stats: ", "Min: ", min(risks), "Max: ", max(risks), 
          "Mean: ", statistics.mean(risks), 
          "Median: ", statistics.median(risks),
          "Total: ", sum(risks))
    print("Inputs stats: ", "Min: ", min(inputs), "Max: ", max(inputs), 
          "Mean: ", statistics.mean(inputs), 
          "Median: ", statistics.median(inputs))

    #############################################################
    ###                   Process Data                        ###
    #############################################################

    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []
    sars_data = []
    for i in range(len(path)-1):
        heat_map = get_radar_heat_map(path[i], radar_locs, img_size, 
                                 aircraft_detection_range, grid_size, radar_radius)
        observations.append({'heat_map': heat_map, 
                            'goal_direction': center_goal(path[i], goal_location),
                            'time_spent': np.exp(i/time_scaling)})
        
        heat_map_next = get_radar_heat_map(path[i+1], radar_locs, img_size, 
                                 aircraft_detection_range, grid_size, radar_radius)
        next_observations.append({'heat_map': heat_map_next, 
                            'goal_direction': center_goal(path[i+1], goal_location),
                            'time_spent': np.exp((i+1)/time_scaling)})

        actions.append(inputs[i])
        dist_to_goal = np.linalg.norm(goal_location - path[i][:2])
        if i >= time_max and dist_to_goal > goal_tolerance:
            rewards.append(offline_data_config['time_limit_penalty'])
            truncations.append(False)
            terminations.append(True)
            break
        elif i < time_max and dist_to_goal <= goal_tolerance:
            rewards.append(offline_data_config['goal_reward'])
            truncations.append(False)
            terminations.append(True)
            print("Breaking...")
            break
        elif i < time_max and dist_to_goal > goal_tolerance:
            rewards.append(-risks[i] - np.exp(i/time_scaling) + 1 / ( dist_to_goal  + 1e-3 ) )
            truncations.append(False)
            terminations.append(False)
        else:
            ### This part shouldn't be called.
            print("Weird call...")
            rewards.append(-np.inf)
            truncations.append(False)
            terminations.append(True)
            break
        
    # print(len(observations))
    # print(len(actions))
    # print(len(rewards))
    # print(len(terminations))
    # print(len(truncations))
    for i in range(len(observations)):
        sars_data.append({'observation': observations[i],
                          'next_observation': next_observations[i],
                          'action': actions[i],
                          'reward': rewards[i],
                          'termination': terminations[i]})

    return sars_data

import pickle 

dataset = None
offline_dataset_size = 0
for i in range(data_num):
    print("Data: ", i)
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

### Save the config
with open(offline_data_config['output_path'] + 'config.yml', 'w') as outfile:
    yaml.dump(offline_data_config, outfile)