# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
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

import pickle
from tensordict import MemoryMappedTensor, TensorDict
from actor_utils import FeatureExtractor, ActorNet
from gym_env.env.radar_env import RadarEnv, MapConfig

TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def is_goal_reached(reward: float, info: Dict) -> bool:
    return info["goal_achieved"]


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset()
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


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
                                           "goal_direction": states[i]['goal_direction'] / self._scaling}, [])
            
            next_states_tensor[i] = TensorDict({"heat_map": next_states[i]['heat_map'], 
                                           "goal_direction": next_states[i]['goal_direction'] / self._scaling}, [])

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

    def add_transition(self, observation, action, reward, next_observation, real_done):
        data = {'observation': observation,
                'next_observation': next_observation,
                'action': action,
                'reward': reward,
                'termination': real_done}
        with open(self._data_path + f'data_{self._buffer_size}.pkl', 'wb') as f:
            pickle.dump(data, f)
            self._buffer_size += 1



def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config['info']["project"],
        name=config['info']["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class TwinQ(nn.Module):
    def __init__(
        self, observation_space, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        dims = [self.feature_extractor.features_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, observations, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.feature_extractor(observations)
        # print("Feature dim: ", feature.shape)
        # print("Action shape: ", action.shape)
        sa = torch.cat([feature, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, observations, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(observations, action))


class ValueFunction(nn.Module):
    def __init__(self, observation_space, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        dims = [self.feature_extractor.features_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, observations) -> torch.Tensor:
        return self.v(self.feature_extractor(observations))


class ImplicitQLearning:
    def __init__(
        self,
        # max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        # self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network


        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        if actor_optimizer is not None:
            self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        pi, policy_out = self.actor(observations)
        
        log_prob = policy_out.distribution.log_prob(actions)
        # action_log_prob = dist.log_prob(pi)
        # print("Actions log probability: ", dist.distribution.log_prob(pi))
        bc_losses = 0
        for i in range(actions.shape[0]):
            bc_losses += - torch.exp( -(pi[i, 0] - actions[i])**2 )*log_prob[i, 0] - torch.exp( -(pi[i, 1] - actions[i])**2 )*log_prob[i, 1]

        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        

    def train(self, batch, update_actor) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch

        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)

        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        if update_actor:
            self._update_policy(adv, observations, actions, log_dict)


        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        if self.actor_optimizer is None:
            return {
                "qf": self.qf.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "vf": self.vf.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "total_it": self.total_it,
            }

        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }


    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

def init_weights(module: nn.Module, gain: float = 1) -> None:
    """
    Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def train(config):
    ##################################################################
    ####      Set up check point path and save the training config ###
    ##################################################################
    checkpoint_path = config['train']['checkpoints_path']
    print(f"Checkpoints path: ", checkpoint_path)
    run_name = config['info']['name'] + '-' + str(uuid.uuid4())[:8]
    os.makedirs(os.path.join(checkpoint_path, run_name), exist_ok=True)
    config['name'] = run_name
    with open(os.path.join(os.path.join(checkpoint_path, run_name), "config.yaml"), "w") as f:
        yaml.dump(config, f)
    #######################################
    ###           Set up Dataloader     ###
    #######################################
    action_dim = config['env']['action_dim']
    action_normalized = config['env']['action_normalized']
    action_scale = config['env']['action_max']
    map_range = config['env']['map_size']
    batch_size = config['train']['batch_size']
    buffer_size = config['train']['buffer_size']
    data_path = config['train']['data_path']
    device = config['train']['device']
    
    replay_buffer = ReplayBuffer(
        action_dim=action_dim,
        action_normalized=action_normalized,
        action_scale=action_scale,
        map_range=map_range,
        buffer_size=buffer_size,
        data_path = data_path,
        device=device,
    )

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
    
    ### Set up Gym Env
    map_config = MapConfig
    map_config.V = config['env']['V']
    map_config.action_dim = action_dim
    map_config.action_max = action_max
    map_config.action_min = action_min
    map_config.action_normalized = action_normalized

    map_config.aircraft_detection_range = config['env']['aircraft_detection_range']
    map_config.delta_t = config['env']['delta_t']
    map_config.goal_tolerance = config['env']['goal_tolerance']
    map_config.img_size = config['env']['img_size']
    map_config.map_size = config['env']['map_size']
    map_config.max_time_step = 10.0*config['env']['max_time_step']
    map_config.goal_direction_normalized = config['env']['observation']['goal_direction_normalized']
    map_config.heat_map_normalized = config['env']['observation']['heat_map_normalized']
    map_config.radar_radius = config['env']['radar_radius']

    env = RadarEnv(map_config, device)
    eval_env = RadarEnv(map_config, device)
    is_env_with_goal = True
    max_steps = map_config.max_time_step 

    # Set up neural network modules.
    pretrained_model_path = config['train']['load_model']
    model_dict = torch.load(pretrained_model_path)
    actor = ActorNet(observation_space, action_space, config['policy']['actor_net_hidden'], hidden_act=nn.Tanh).to(device)
    actor.load_state_dict(model_dict['actor'])
    print("Load pretrained actor from: ", pretrained_model_path)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad = False
    q_network = TwinQ(observation_space, action_dim).to(device)
    init_weights(q_network)
    v_network = ValueFunction(observation_space).to(device)
    init_weights(v_network)
    
    actor_lr = config['train']['actor_lr']
    vf_lr = config['train']['vf_lr']
    qf_lr = config['train']['qf_lr']
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=qf_lr)
    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    max_steps = config['train']['max_timesteps']
    kwargs = {
        # "max_action": max_action,
        "actor": actor,
        "actor_optimizer": None,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config['train']['discount'],
        "tau": config['train']['tau'],
        "device": device,
        # IQL
        "beta": config['train']['beta'],
        "iql_tau": config['train']['iql_tau'],
        "max_steps": max_steps,
    }

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    wandb_init(config)
    eval_freq = config['eval']['eval_freq']
    n_episodes = config['eval']['n_episodes']
    eval_seed = config['eval']['seed']
    # evaluations = []

    state, done = env.reset()
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    value_networks_warmup_timesteps = config['train']['value_network_warmup_steps']
    max_training_steps = config['train']['max_timesteps']
    for t in range(int(value_networks_warmup_timesteps) + int(max_training_steps)):
        if t == value_networks_warmup_timesteps:
            actor.train()
            for param in actor.parameters():
                param.requires_grad = True
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
            trainer.actor_optimizer = actor_optimizer
            trainer.actor_lr_schedule = CosineAnnealingLR(trainer.actor_optimizer, max_steps)
            print("Start Online Fine-Tuning.")
        online_log = {}
        if t >= value_networks_warmup_timesteps:
            episode_step += 1
            _, policy_distribution = actor(state.unsqueeze(dim=0))
            action = policy_distribution.sample()
            log_prob = policy_distribution.distribution.log_prob(action)
            action_idx = torch.argmax(log_prob, dim=-1, keepdim=True)
            action = torch.clamp(action_max * action, -action_max, action_max)
            action = torch.gather(action, dim=-1, index=action_idx)
            action = action.cpu().data.numpy().flatten()[0]
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward

            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            replay_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state
            if done:
                state, done = env.reset()
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                train_successes.append(goal_achieved)
                online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        batch = replay_buffer.sample(batch_size)
        batch = [b.to(device) for b in batch]
        log_dict = trainer.train(batch, update_actor=bool(t >= value_networks_warmup_timesteps))
        log_dict["value_networks_warmup_iter" if t < value_networks_warmup_timesteps else "online_iter"] = (
            t if t < value_networks_warmup_timesteps else t - value_networks_warmup_timesteps
        )
        log_dict.update(online_log)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=device,
                n_episodes=n_episodes,
                seed=eval_seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            # normalized = eval_env.get_normalized_score(eval_score)
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= value_networks_warmup_timesteps and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            # normalized_eval_score = normalized * 100.0
            # evaluations.append(normalized_eval_score)
            # eval_log["eval/normalized_score"] = normalized_eval_score
            print("---------------------------------------")
            print(
                f"Evaluation over {n_episodes} episodes: "
                f"{eval_score:.3f}"
            )
            print("---------------------------------------")
            torch.save(
                trainer.state_dict(),
                os.path.join(os.path.join(checkpoint_path, run_name), f"checkpoint_{t}.pt"),
            )
            wandb.log(eval_log, step=trainer.total_it)


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Finetune a pretrained policy with IQL.'
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
