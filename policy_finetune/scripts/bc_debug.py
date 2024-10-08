import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
import yaml
import pickle
import actor_utils
from actor_utils import FeatureExtractor, ActorNet
from tensordict import MemoryMappedTensor, TensorDict
import shutil
import pathlib
TensorBatch = List[torch.Tensor]

class ReplayBuffer:
    def __init__(
        self,
        action_dim: int,
        action_normalized: bool,
        action_scale: float,
        map_range: float,
        buffer_size: int,
        data_path: str,
        device: str = "cuda"
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
            "goal_direction": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32)
            # 'time_spent':  MemoryMappedTensor.empty((len(states), 1), dtype=torch.float32) 
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
            "goal_direction": MemoryMappedTensor.empty((len(states), 2), dtype=torch.float32)
            # 'time_spent':  MemoryMappedTensor.empty((len(states), 1), dtype=torch.float32) 
        },
            batch_size=[batch_size],
            device=self._device,
        )

        for i in range(batch_size):
            states_tensor[i] = TensorDict({"heat_map": states[i]['heat_map'], 
                                           "goal_direction": states[i]['goal_direction'] / self._scaling}, [])
            
            next_states_tensor[i] = TensorDict({"heat_map": next_states[i]['heat_map'], 
                                           "goal_direction": next_states[i]['goal_direction'] / self._scaling}, [])
            # states_tensor[i] = TensorDict({"heat_map": states[i]['heat_map'], 
            #                                "goal_direction": states[i]['goal_direction'] / self._scaling,
            #                                'time_spent': states[i]['time_spent']}, [])
            
            # next_states_tensor[i] = TensorDict({"heat_map": next_states[i]['heat_map'], 
            #                                "goal_direction": next_states[i]['goal_direction'] / self._scaling,
            #                                'time_spent': next_states[i]['time_spent']}, [])

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

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError



def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, observation_space, action_dim, action_max=1, action_min=-1, num_modes=2):
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
    

class BC:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch
        batch_size = len(state)
        # print("Batch Size: ", batch_size)
        if batch_size < 1:
            print("Batch Size: ", batch_size)
        # Compute actor loss
        pi, dist = self.actor(state)
        # print(dist)
        # print("Actions: ", pi)
        actor_loss = 0
        log_prob = dist.distribution.log_prob(pi)
        # action_log_prob = dist.log_prob(pi)
        # print("Actions log probability: ", dist.distribution.log_prob(pi))
        for i in range(pi.shape[0]):
            actor_loss += - torch.exp( -(pi[i, 0] - action[i])**2 )*log_prob[i, 0] - torch.exp( -(pi[i, 1] - action[i])**2 )*log_prob[i, 1]

        if batch_size > 0:
            actor_loss /= batch_size
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]

def init_weights(module: nn.Module, gain: float = 1) -> None:
    """
    Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


# @pyrallis.wrap()
def train(config):
    checkpoint_path = config['train']['checkpoints_path']
    print(f"Checkpoints path: ", checkpoint_path)
    run_name = config['name'] + '-' + str(uuid.uuid4())[:8]
    os.makedirs(os.path.join(checkpoint_path, run_name), exist_ok=True)
    config['name'] = run_name
    with open(os.path.join(os.path.join(checkpoint_path, run_name), "config.yaml"), "w") as f:
        yaml.dump(config, f)

    ### Saving this file and the gym env for reproducibility.
    print("Saving settings...")
    shutil.copy(os.path.(__file__), os.path.join(checkpoint_path, run_name))
    shutil.copy(os.path.abspath(actor_utils.__file__), os.path.join(checkpoint_path, run_name))

    # Set environment parameters.
    img_size = config['env']['img_size']
    observation_img_size = [1, img_size, img_size]
    action_dim = config['env']['action_dim']
    map_range = config['env']['map_size']
    action_normalized = config['env']['action_normalized']
    if action_normalized:
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))
    else:
        action_min = config['env']['action_min']
        action_max = config['env']['action_max']
        action_space = gym.spaces.Box(low=action_min, high=action_max, shape=(action_dim,))

    buffer_size = config['train']['buffer_size']
    device = config['train']['device']
    data_path = config['train']['data_path']
    action_scale = config['env']['action_max']
    replay_buffer = ReplayBuffer(
        action_dim,
        action_normalized,
        action_scale,
        map_range,
        buffer_size,
        data_path,
        device
    )

    observation_space= gym.spaces.Dict({"heat_map": gym.spaces.Box(0, 255, observation_img_size), 
                            "goal_direction": gym.spaces.Box(-1, 1, shape=(2,))})
    # observation_space = gym.spaces.Box(-1, 1, shape=(5,))

    assert config['policy']['hidden_act'] == 'Tanh' or config['policy']['hidden_act'] == 'ReLU', "Currently only support ReLU or Tanh."
    if config['policy']['hidden_act'] == 'Tanh':
        actor = ActorNet(observation_space, action_space, config['policy']['actor_net_hidden'], hidden_act=nn.Tanh).to(device)
    if config['policy']['hidden_act'] == 'ReLU':
        actor = ActorNet(observation_space, action_space, config['policy']['actor_net_hidden'], hidden_act=nn.ReLU).to(device)
    
    load_model_from = config['train']['load_model']
    if load_model_from == '':
        init_weights(actor)
    else:
        print("Load model from: ", load_model_from)
        model_dict = torch.load(load_model_from)
        actor.load_state_dict(model_dict['actor'])
        actor.train()

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['policy']['lr_rate'])

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config['train']['discount'],
        "device": device,
    }

    # Initialize policy
    trainer = BC(**kwargs)

    if config['train']['load_model'] != "":
        policy_file = Path(config['train']['load_model'] )
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(config)

    max_timesteps = config['train']['max_timesteps']
    batch_size = config['train']['batch_size']
    eval_freq = config['train']['eval_freq']

    for t in range(int(max_timesteps)):
        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]
        log_dict = trainer.train(batch)
        if t % 500 == 0:
            wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if t % eval_freq == 0:
            print(f"Time steps: {t + 1}")
            if checkpoint_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(checkpoint_path, run_name, f"checkpoint_{t}.pt"),
                )


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a policy with BC.'
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
