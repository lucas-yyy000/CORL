import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import yaml
import argparse
# import d4rl
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import pickle
from tensordict import MemoryMappedTensor, TensorDict
from actor_utils import FeatureExtractor, DiagGaussianDistribution
from gym_env.env.radar_env import RadarEnv, MapConfig

class Dataset:
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

        self._device = device
        self._data_path = data_path

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._buffer_size, size=batch_size)

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
            'time_spent':  MemoryMappedTensor.empty((len(states), 1), dtype=torch.float32) 
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
            'time_spent':  MemoryMappedTensor.empty((len(states), 1), dtype=torch.float32) 
        },
            batch_size=[batch_size],
            device=self._device,
        )

        for i in range(batch_size):
            states_tensor[i] = TensorDict({"heat_map": states[i]['heat_map'], 
                                           "goal_direction": states[i]['goal_direction'] / self._scaling,
                                           'time_spent': states[i]['time_spent']}, [])
            
            next_states_tensor[i] = TensorDict({"heat_map": next_states[i]['heat_map'], 
                                           "goal_direction": next_states[i]['goal_direction'] / self._scaling,
                                           'time_spent': next_states[i]['time_spent']}, [])

        if self._action_normalized:
            if self._action_dim == 1:
                actions = self._to_tensor(torch.from_numpy(np.asarray(actions) / self._action_scale)).unsqueeze(dim=-1)
            else:
                actions = self._to_tensor(torch.from_numpy(np.asarray(actions) / self._action_scale))
        else:
            actions = self._to_tensor(torch.from_numpy(np.asarray(actions)))
        rewards = self._to_tensor(torch.from_numpy(np.asarray(rewards)))
        dones = self._to_tensor(torch.from_numpy(np.asarray(dones)))

        return [states_tensor, actions, rewards, next_states_tensor, dones]

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config['info']["project"],
        name=config['info']["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

class ActorAdaptationNet(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        in_size = self.feature_extractor.features_dim
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(in_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU()
        )
        self.act_dist = DiagGaussianDistribution(3*2)
        self.action_net, self.log_std = self.act_dist.proba_distribution_net(self.latent_dim)

    def forward(self, observations, deterministic=False) -> torch.Tensor:
        latent = self.mlp(self.feature_extractor(observations))
        mean_action = self.action_net.forward(latent)
        distribution = self.act_dist.proba_distribution(mean_action, self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        actions = distribution.get_actions(deterministic=deterministic)

        return actions, distribution
    
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
    device = config['train']['device']

    # Set environment parameters.
    img_size = config['env']['img_size']
    observation_img_size = [1, img_size, img_size]
    if action_normalized:
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2*action_dim,))
        action_min = -1.0
        action_max = 1.0
    else:
        action_min = config['env']['action_min']
        action_max = config['env']['action_max']
        action_space = gym.spaces.Box(low=action_min, high=action_max, shape=(2*action_dim,))
    
    observation_space= gym.spaces.Dict({"heat_map": gym.spaces.Box(0, 255, observation_img_size), 
                        "goal_direction": gym.spaces.Box(-1, 1, shape=(2,))})
    observation_space_res = gym.spaces.Dict({"heat_map": gym.spaces.Box(0, 255, observation_img_size), 
                        "goal_direction": gym.spaces.Box(-1, 1, shape=(2,)),
                        "base_waypoints": gym.spaces.Box(-1, 1, shape=(6,))
                        })
    
    batch_size = config['train']['batch_size']
    num_iterations = config['train']['max_training_iterations']


    ### Set up neural network modules.
    pretrained_model_path = config['train']['load_model']
    model_dict = torch.load(pretrained_model_path)
    actor = ActorNet(observation_space, action_space, config['train']['policy_specs']['actor_net_hidden'], hidden_act=nn.ReLU).to(device)
    actor.load_state_dict(model_dict['actor'])
    print("Load pretrained actor from: ", pretrained_model_path)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad = False

    res_actor = ActorAdaptationNet(observation_space_res)
    res_actor_lr = config['train']['res_actor_lr']
    res_actor_optimizer = torch.optim.Adam(res_actor.parameters(), lr=res_actor_lr)
    res_actor_lr_schedule = CosineAnnealingLR(res_actor_optimizer, num_iterations)

    for iteration in range(1, num_iterations + 1):
        

        if iteration % 500 == 0:
            ### Log with wandb
            log_dict = {}
            log_dict['actor_learning_rate'] = res_actor_optimizer.param_groups[0]["lr"]
            log_dict['policy_loss'] = pg_loss.item()
            wandb.log(log_dict, step=iteration)

        if iteration % 10_000 == 0:
            print("Iteration: ", iteration)
            torch.save(
                actor.state_dict(),
                os.path.join(os.path.join(checkpoint_path, run_name), f"actor_checkpoint_{iteration}.pt"),
            )


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Finetune a pretrained bc policy.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    with open(args.config,"r") as file_object:
        config = yaml.load(file_object,Loader=yaml.SafeLoader)

    train(config)