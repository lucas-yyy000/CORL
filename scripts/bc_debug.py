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
<<<<<<< Updated upstream
from actor_utils import FeatureExtractor, ActorNet
=======
from actor_utils import DiagGaussianDistribution, ActorNet
>>>>>>> Stashed changes
from tensordict import MemoryMappedTensor, TensorDict

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
<<<<<<< Updated upstream

=======
            
        # states = 
        # observations = []
        # next_observations = []
        # for i in range(batch_size):
        #     state = states[i]
        #     state[0] /= self._scaling
        #     state[1] /= self._scaling
        #     state[3] /= self._scaling
        #     state[4] /= self._scaling
        #     observations.append(state)

        #     next_state = next_states[i]
        #     next_state[0] /= self._scaling
        #     next_state[1] /= self._scaling
        #     next_state[3] /= self._scaling
        #     next_state[4] /= self._scaling
        #     next_observations.append(next_state)

        # observations = self._to_tensor(torch.from_numpy(np.asarray(observations)))
        # next_observations = self._to_tensor(torch.from_numpy(np.asarray(next_observations)))
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
    
=======
>>>>>>> Stashed changes
# class ActorNet(nn.Module):
#     def __init__(self,
#                  observation_space,
#                  action_space,
#                  hidden_sizes,
#                  hidden_act=nn.ReLU):
#         super().__init__()
#         if not isinstance(hidden_sizes, list):
#             raise TypeError('hidden_sizes should be a list')
#         self.action_space = action_space
#         action_dim = action_space.shape[0]
<<<<<<< Updated upstream
#         self.feature_extractor = FeatureExtractor(observation_space)
#         in_size = self.feature_extractor.features_dim
=======
#         in_size = observation_space.shape[0]
>>>>>>> Stashed changes
#         mlp_extractor : List[nn.Module] = []
#         for curr_layer_dim in hidden_sizes:
#             mlp_extractor.append(nn.Linear(in_size, curr_layer_dim))
#             mlp_extractor.append(hidden_act())
#             in_size = curr_layer_dim

<<<<<<< Updated upstream
#         self.latent_dim = in_size
#         mlp_extractor.append(nn.Linear(self.latent_dim, action_dim))
#         mlp_extractor.append(nn.Tanh())
#         self.policy_net = nn.Sequential(*mlp_extractor)


#     def forward(self, observations, deterministic=False):
#         feature = self.feature_extractor(observations)
#         actions = self.policy_net.forward(feature)

#         return actions
=======
#         # mlp_extractor.append(nn.Linear(in_size, action_dim))
#         # mlp_extractor.append(nn.Tanh())
#         # # self.latent_dim = in_size
#         self.policy_net = nn.Sequential(*mlp_extractor)
#         self.act_dist = DiagGaussianDistribution(action_dim)
#         self.action_net, self.log_std = self.act_dist.proba_distribution_net(in_size)

#     def forward(self, observations, deterministic=False):
#         # feature = self.feature_extractor(observations)
#         # latent = self.policy_net.forward(feature)
#         # mean_action = self.action_net.forward(latent)
#         # distribution = self.act_dist.proba_distribution(mean_action, self.log_std)
#         # actions = distribution.get_actions(deterministic=deterministic)
#         # # log_prob = distribution.log_prob(actions)
#         # actions = actions.reshape((-1, *self.action_space.shape)) 
#         mean_action = self.policy_net(observations)
#         distribution = self.act_dist.proba_distribution(mean_action, self.log_std)
#         actions = distribution.get_actions(deterministic=deterministic)

#         return actions, distribution
>>>>>>> Stashed changes
    

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

<<<<<<< Updated upstream
    def train(self, batch) -> Dict[str, float]:
=======
    # def train(self, batch: TensorBatch) -> Dict[str, float]:
    #     log_dict = {}
    #     self.total_it += 1

    #     state, action, _, _, _ = batch

    #     pi, dist = self.actor(state)
    #     actor_loss = 0
    #     for i in range(pi.shape[0]):
    #         actor_loss += - torch.exp( -(pi[i, 0] - action[i])**2 )*dist.log_prob(pi)[0] - torch.exp( -(pi[i, 1] - action[i])**2 )*dist.log_prob(pi)[1]
    #     actor_loss /= pi.shape[0]

    #     # # Compute actor loss
    #     # pi, _ = self.actor(state)
    #     # actor_loss = F.mse_loss(pi, action)
    #     log_dict["actor_loss"] = actor_loss.item()
    #     # Optimize the actor
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()

    #     return log_dict

    def train(self, batch: TensorBatch) -> Dict[str, float]:
>>>>>>> Stashed changes
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch
<<<<<<< Updated upstream
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
=======


        # Compute actor loss
        pi, dist = self.actor(state)

        actor_loss = 0
        log_prob = dist.distribution.log_prob(pi)
        # print("Log Probability shape: ", log_prob.shape)
        prob = torch.exp(log_prob)
        # print("Probability shape: ", prob.shape)
        # total_w = 0
        for i in range(pi.shape[0]):
            w1 = prob[i, 0]*torch.exp( -(pi[i, 0] - action[i])**2 )
            w2 = prob[i, 1]*torch.exp( -(pi[i, 1] - action[i])**2 )
            w = 0.5*(w1 + w2 + 1e-6)
            actor_loss += - (w1 / w)*log_prob[i, 0] - (w2 / w)*log_prob[i, 1]

        if pi.shape[0] == 0:
            print("Actions have suspicious shape: ", pi.shape)
        actor_loss /= pi.shape[0]
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
=======
    # observation_space= gym.spaces.Dict({"heat_map": gym.spaces.Box(0, 255, observation_img_size), 
    #                         "goal_direction": gym.spaces.Box(-1, 1, shape=(2,)),
    #                         'time_spent': gym.spaces.Box(low=1.0, high=np.inf, shape=(1,))})
>>>>>>> Stashed changes
    observation_space= gym.spaces.Dict({"heat_map": gym.spaces.Box(0, 255, observation_img_size), 
                            "goal_direction": gym.spaces.Box(-1, 1, shape=(2,))})
    # observation_space = gym.spaces.Box(-1, 1, shape=(5,))

    assert config['policy']['hidden_act'] == 'Tanh' or config['policy']['hidden_act'] == 'ReLU', "Currently only support ReLU or Tanh."
    if config['policy']['hidden_act'] == 'Tanh':
        actor = ActorNet(observation_space, action_space, config['policy']['actor_net_hidden'], hidden_act=nn.Tanh).to(device)
    if config['policy']['hidden_act'] == 'ReLU':
        actor = ActorNet(observation_space, action_space, config['policy']['actor_net_hidden'], hidden_act=nn.ReLU).to(device)
<<<<<<< Updated upstream
    
=======
>>>>>>> Stashed changes
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
