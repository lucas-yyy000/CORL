import os
import random
import time
from dataclasses import dataclass
import uuid

import gym
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
import argparse
import yaml
import pickle
from policy_finetune.scripts.actor_utils import FeatureExtractor
from tensordict import MemoryMappedTensor, TensorDict
from gym_env.env.radar_env import RadarEnv, MapConfig

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config['info']["project"],
        name=config['info']["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def ordereddict_to_tensordict(dict):
    td = TensorDict({
        'heat_map': torch.from_numpy(dict['heat_map']).float(),
        'goal_direction': torch.from_numpy(dict['goal_direction']).float(),
        'current_loc': torch.from_numpy(dict['current_loc']).float(),
        'time_spent': torch.from_numpy(dict['time_spent']).float(),
        'risk_measure': torch.from_numpy(dict['risk_measure']).float()
    }, batch_size=[dict['goal_direction'].shape[0]])
    return td

class ReplayBufferSamples():
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer:
    def __init__(
        self,
        action_dim: int,
        action_normalized: bool,
        action_scale: float,
        map_range: float,
        buffer_size: int,
        data_path: str,
        device: str
    ):  
        self._action_dim = action_dim
        self._action_normalized = action_normalized
        self._action_scale = action_scale
        self._scaling = map_range / 2.0

        self._buffer_size = buffer_size
        self._new_data_idx = buffer_size
        self._max_buffer_size = 1_000_000
        self._demonstration_size = buffer_size

        self._device = device
        self._data_path = data_path

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int):
        ### Symmetric Sampling
        demonstration_indices = np.random.randint(0, self._demonstration_size, size=int(batch_size/2))
        replay_buffer_indices = np.random.randint(self._demonstration_size, self._buffer_size, size=int(batch_size/2))
        indices = np.concatenate((demonstration_indices, replay_buffer_indices))

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            with open(self._data_path + f'data_{indices[i]}.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
                states.append(loaded_dict['observation'])
                actions.append(loaded_dict['action'])
                rewards.append(loaded_dict['reward'])
                next_states.append(loaded_dict['next_observation'])
                dones.append(loaded_dict['termination'])


        states_tensor = TensorDict(
        {
            "heat_map": MemoryMappedTensor.empty(
                (len(states), *states[0]['heat_map'].shape),
                dtype=torch.float32,
            ),
            "goal_direction": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32),
            "current_loc": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32),
            "time_spent":  MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32),
            "risk_measure": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32)
        },
            batch_size=[batch_size],
            device=self._device,
        )

        next_states_tensor = TensorDict(
        {
            "heat_map": MemoryMappedTensor.empty(
                (len(states), *next_states[0]['heat_map'].shape),
                dtype=torch.float32,
            ),
            "goal_direction": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32),
            "current_loc": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32),
            "time_spent":  MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32),
            "risk_measure": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32)
        },
            batch_size=[batch_size],
            device=self._device,
        )

        for i in range(batch_size):
            states_tensor[i] = TensorDict({"heat_map": states[i]['heat_map'], 
                                           "goal_direction": states[i]['goal_direction'] / self._scaling,
                                           "current_loc": states[i]['current_loc'] / (2.0*self._scaling),
                                           "time_spent": states[i]['time_spent'],
                                           "risk_measure": states[i]['risk_measure']}, [])
            
            next_states_tensor[i] = TensorDict({"heat_map": next_states[i]['heat_map'], 
                                           "goal_direction": next_states[i]['goal_direction'] / self._scaling,
                                           "current_loc": states[i]['current_loc'] / (2.0*self._scaling),
                                           "time_spent": states[i]['time_spent'],
                                            "risk_measure": states[i]['risk_measure']}, [])

        if self._action_normalized:
            if self._action_dim == 1:
                actions = self._to_tensor(torch.from_numpy(np.asarray(actions) / self._action_scale)).unsqueeze(dim=-1)
            else:
                actions = self._to_tensor(torch.from_numpy(np.asarray(actions) / self._action_scale))
        else:
            actions = self._to_tensor(torch.from_numpy(np.asarray(actions)))
        rewards = self._to_tensor(torch.from_numpy(np.asarray(rewards)))
        dones = self._to_tensor(torch.from_numpy(np.asarray(dones)))

        samples = ReplayBufferSamples()
        samples.observations = states_tensor
        samples.actions = actions
        samples.rewards = rewards
        samples.next_observations = next_states_tensor
        samples.dones = dones
        return samples

    def add_transition(self, observation, action, reward, next_observation, real_done):
        observation = ordereddict_to_tensordict(observation)
        next_observation = ordereddict_to_tensordict(next_observation)
        # print("Add observation: ", observation[0])
        # print("Add action: ", action)
        # print("Add reward: ", reward)
        for i in range(action.shape[0]):
            data = {'observation': observation[i].to_dict(),
                    'next_observation': next_observation[i].to_dict(),
                    'action': action[i][0],
                    'reward': reward[i],
                    'termination': real_done[i]}
            if self._new_data_idx >= self._max_buffer_size:
                self._new_data_idx = self._demonstration_size
            with open(self._data_path + f'data_{self._new_data_idx}.pkl', 'wb') as f:
                pickle.dump(data, f)
                self._new_data_idx += 1
                if self._buffer_size < self._max_buffer_size:
                    self._buffer_size += 1
    
# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, observation_space, action_dim):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        in_size = self.feature_extractor.features_dim
        self.fc1 = nn.Linear(in_size + action_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, observation, a):
        x = self.feature_extractor(observation)
        x = torch.cat([x, a], 1)
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, observation_space, action_dim, action_max, action_min):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        in_size = self.feature_extractor.features_dim
        self.fc1 = nn.Linear(in_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, action_dim)
        self.fc_logstd = nn.Linear(64, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_max - action_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_max + action_min) / 2.0, dtype=torch.float32)
        )

    def forward(self, observation):
        x = self.feature_extractor(observation)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, observation):
        mean, log_std = self(observation)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    

def train(config):
    ##################################################################
    ####      Set up check point path and save the training config ###
    ##################################################################
    checkpoint_path = config['train']['checkpoints_path']
    print(f"Checkpoints path: ", checkpoint_path)
    run_name = config['info']['name'] + '-' + str(uuid.uuid4())[:8]
    os.makedirs(os.path.join(checkpoint_path, run_name), exist_ok=True)
    config['name'] = run_name
    print("Run name: ", run_name)
    with open(os.path.join(os.path.join(checkpoint_path, run_name), "config.yaml"), "w") as f:
        yaml.dump(config, f)

    #######################################
    ###      Set up Environment    ###
    #######################################
    action_dim = config['env']['action_dim']
    action_normalized = config['env']['action_normalized']
    batch_size = config['train']['batch_size']
    device = config['train']['device']

    # Set environment parameters.
    img_size = config['env']['img_size']
    observation_img_size = [1, img_size, img_size]
    if action_normalized:
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))
        action_min = -1.0
        action_max = 1.0
    else:
        action_min = config['env']['action_min']
        action_max = config['env']['action_max']
        action_space = gym.spaces.Box(low=action_min, high=action_max, shape=(action_dim,))
    
    observation_space= gym.spaces.Dict({"heat_map": gym.spaces.Box(0, 255, observation_img_size), 
                        "goal_direction": gym.spaces.Box(-1, 1, shape=(2,)),
                        "current_loc": gym.spaces.Box(0, 1, shape=(2,)),
                        "time_spent": gym.spaces.Box(-1, 1, shape=(2,)),
                        "risk_measure": gym.spaces.Box(-1, 1, shape=(2,))})


    actor = Actor(observation_space, action_dim, action_max, action_min).to(device)
    qf1 = SoftQNetwork(observation_space, action_dim).to(device)
    qf2 = SoftQNetwork(observation_space, action_dim).to(device)
    qf1_target = SoftQNetwork(observation_space, action_dim).to(device)
    qf2_target = SoftQNetwork(observation_space, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_lr = config['train']['q_lr']
    policy_lr = config['train']['policy_lr']
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

    ### Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=q_lr)

    ### Set up replay buffer
    action_scale = config['env']['action_max']
    map_range = config['env']['map_size']
    buffer_size = config['train']['buffer_size']
    data_path = config['train']['data_path']
    replay_buffer = ReplayBuffer(
        action_dim=action_dim,
        action_normalized=action_normalized,
        action_scale=action_scale,
        map_range=map_range,
        buffer_size=buffer_size,
        data_path = data_path,
        device=device,
    )

    ### Set up Gym Env
    map_config = MapConfig
    map_config.V = config['env']['V']
    map_config.action_dim = action_dim
    map_config.action_max = action_max
    map_config.action_min = action_min
    map_config.action_normalized = action_normalized
    map_config.reward_normalization_factor = config['env']['reward_normalization_factor']
    map_config.aircraft_detection_range = config['env']['aircraft_detection_range']
    map_config.delta_t = config['env']['delta_t']
    map_config.goal_tolerance = config['env']['goal_tolerance']
    map_config.img_size = config['env']['img_size']
    map_config.map_size = config['env']['map_size']
    map_config.max_time_step = config['env']['max_time_step']
    map_config.goal_direction_normalized = config['env']['observation']['goal_direction_normalized']
    map_config.heat_map_normalized = config['env']['observation']['heat_map_normalized']
    map_config.radar_radius = config['env']['radar_radius']
    map_config.interceptor_launch_time = config['env']['interceptor_launch_time']
    map_config.interceptor_abort_time = config['env']['interceptor_abort_time']

    num_envs = config['train']['num_envs']
    envs = gym.vector.AsyncVectorEnv([
        lambda: RadarEnv(MapConfig(), 'cuda')
        for _ in range(num_envs)
    ])
    # print("Envs: ", envs.observation_space)
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    wandb_init(config)
    
    ### Load training specs
    total_timesteps = config['train']['total_timesteps']
    learning_starts = config['train']['learning_starts']
    policy_frequency = config['train']['policy_frequency']
    target_network_frequency = config['train']['target_network_frequency']
    tau = config['train']['tau']
    gamma = config['train']['gamma']

        
    def lambda_lr(time_step):
        return 1 - 0.7*(time_step / total_timesteps)
    q_scheduler = optim.lr_scheduler.LambdaLR(q_optimizer, lr_lambda=lambda_lr)
    a_scheduler = optim.lr_scheduler.LambdaLR(a_optimizer, lr_lambda=lambda_lr)


    for global_step in range(total_timesteps):
        if global_step % 500 == 0:
            print("Global Steps: ", global_step)
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:
            actions = np.array([action_space.sample() for _ in range(num_envs)])
            # print("Actions shape: ", actions.shape)
        else:
            actions, _, _ = actor.get_action(ordereddict_to_tensordict(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        replay_buffer.add_transition(obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            
            data = replay_buffer.sample(batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % 500 == 0:
                print("Logging Q loss.")
                wandb.log({'q_loss': qf_loss.item()}, step=global_step)

            if global_step % policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if global_step % 500 == 0:
                        print("Logging actor loss.")
                        wandb.log({'actor_loss': actor_loss.item()}, step=global_step)
                    # Autotune
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        ### Update learning rates for Q and alpha.
        q_scheduler.step()
        a_scheduler.step()

        if global_step % 10_000 == 0:
            print("Iteration: ", global_step)
            torch.save(
                qf1.state_dict(),
                os.path.join(os.path.join(checkpoint_path, run_name), f"q1_checkpoint_{global_step}.pt"),
            )
            torch.save(
                qf2.state_dict(),
                os.path.join(os.path.join(checkpoint_path, run_name), f"q2_checkpoint_{global_step}.pt"),
            )
            torch.save(
                actor.state_dict(),
                os.path.join(os.path.join(checkpoint_path, run_name), f"actor_checkpoint_{global_step}.pt"),
            )

    envs.close()

def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Efficient Online Training from offline data (SAC).'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.config,"r") as file_object:
        config = yaml.load(file_object,Loader=yaml.SafeLoader)
    torch.multiprocessing.set_start_method('spawn')
    train(config)