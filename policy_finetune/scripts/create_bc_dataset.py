import numpy as np
from policy_finetune.scripts.utils import *
import statistics
import yaml
from policy_finetune.scripts.simulate_detection import DetectionSimulator
from evasion_guidance.scripts.evasion_risk import EvasionRisk

data_config_path = "/home/yixuany/workspace/evasion_nonlinear_guidance/evasion_guidance/data/config.yml"
with open(data_config_path,"r") as file_object:
    data_config = yaml.load(file_object,Loader=yaml.SafeLoader)


data_path =  "/home/yixuany/workspace/CORL/data/"

map_range = data_config['env']['map_range']
# radar_radius = data_config['env']['radar_radius']
min_num_radar = data_config['env']['min_num_radar']
max_num_radar = data_config['env']['max_num_radar']
V = data_config['planner']['V']
L1 = data_config['planner']['L1']
time_interval = data_config['planner']['delta_t']
risk_interval = data_config['planner']['risk_buffer_length']

with open("/home/yixuany/workspace/CORL/policy_finetune/params/offline_data_config.yaml","r") as file_object:
    offline_data_config = yaml.load(file_object,Loader=yaml.SafeLoader)

processed_data_path = offline_data_config['output_path']
data_num = offline_data_config['load_data_num']
img_size = offline_data_config['env']['img_size']
radar_radius = offline_data_config['env']['radar_radius']
aircraft_detection_range = offline_data_config['env']['aircraft_detection_range']
grid_size=2*aircraft_detection_range/img_size
time_max = offline_data_config['env']['max_time_step']
observation_img_size = [1, img_size, img_size]
goal_tolerance = offline_data_config['goal_tolerance']
time_scaling = offline_data_config['env']['time_scaling']
interceptor_launch_time = offline_data_config['env']['interceptor_launch_time']
interceptor_abort_time = offline_data_config['env']['interceptor_abort_time']

def out_of_bound(loc, eps=50.0):
    x = loc[0]
    y = loc[1]
    out_of_bound = bool(
                x < 0.0 - eps
                or x > map_range + eps
                or y < 0.0 - eps
                or y > map_range + eps
            )
    return out_of_bound

def generate_episode_data(data):
    # init_state = data['start_state']
    goal_location = data['goal_location']
    radar_locs = data['radar_locations']
    # radar_orientations = data['radar_orientations']
    path = data['state_history']
    inputs = data['input_history']
    # risks = data['risk_history']

    #############################################################
    ###       Print some stats about the collected data       ###
    #############################################################

    print("Episode length: ", len(path))
    # print("Risks stats: ", "Min: ", min(risks), "Max: ", max(risks), 
    #       "Mean: ", statistics.mean(risks), 
    #       "Median: ", statistics.median(risks),
    #       "Total: ", sum(risks))
    # print("Inputs stats: ", "Min: ", min(inputs), "Max: ", max(inputs), 
    #       "Mean: ", statistics.mean(inputs), 
    #       "Median: ", statistics.median(inputs))

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

    total_time_in_detection_range = 0
    print("Radar radius: ", radar_radius)
    risk_evaluator = EvasionRisk(radar_locs, risk_interval, radar_radius)
    heat_map_prev = None

    for i in range(len(path)-1):        
        heat_map = get_radar_heat_map(path[i], radar_locs, img_size, 
                                 aircraft_detection_range, grid_size)
        # if heat_map_prev is None:
        #     heat_map_stacked = np.repeat(heat_map, 3, 0)
        # else:
        #     heat_map_stacked = np.roll(heat_map_prev, -1, 0)
        #     heat_map_stacked[-1, :, :] = heat_map
        # heat_map_prev = heat_map_stacked
        goal_dir = center_state(path[i], goal_location)
        observations.append({'heat_map': heat_map, 
                            'goal_direction': goal_dir,
                            'current_loc': path[i][:2],
                            'time_spent': np.array([np.cos(i / time_max), np.sin(i / time_max)])
                            })
        
        heat_map_next = get_radar_heat_map(path[i+1], radar_locs, img_size, 
                                 aircraft_detection_range, grid_size)
        # heat_map_stacked_next = np.roll(heat_map_stacked, -1, 0)
        # heat_map_stacked_next[-1, :, :] = heat_map_next

        next_observations.append({'heat_map': heat_map_next, 
                            'goal_direction': center_state(path[i+1], goal_location),
                            'current_loc': path[i+1][:2],
                            'time_spent': np.array([np.cos(i / time_max), np.sin(i / time_max)])
                            })
        
        actions.append(inputs[i])
        risk = risk_evaluator.evalute_risk(path[i], inputs[i])
        total_time_in_detection_range_prev = total_time_in_detection_range
        if risk > 0:
            total_time_in_detection_range += 1
        else:
            total_time_in_detection_range = 0

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
        elif i < time_max and out_of_bound(path[i]):
            rewards.append(offline_data_config['out_of_bound_penalty'])
            truncations.append(False)
            terminations.append(True)
            print("Out of bound at step: ", i)
            break
        elif i < time_max and dist_to_goal > goal_tolerance:
            reward = - risk - 0.1*total_time_in_detection_range_prev - np.linalg.norm(goal_dir) / (map_range/2.0)
            rewards.append(reward)
            truncations.append(False)
            terminations.append(False)
            prev_goal_dir = goal_dir
        else:
            ### This part shouldn't be called.
            raise Exception("Weird call...")

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

### Save the config
with open(offline_data_config['output_path'] + 'config.yml', 'w') as outfile:
    yaml.dump(offline_data_config, outfile)