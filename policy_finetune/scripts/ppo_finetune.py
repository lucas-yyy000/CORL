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
from actor_utils import FeatureExtractor, ActorNet
from gym_env.env.radar_env import RadarEnv, MapConfig


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


class CriticNet(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.feature_extractor = FeatureExtractor(observation_space)
        in_size = self.feature_extractor.features_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, observations) -> torch.Tensor:
        return self.critic(self.feature_extractor(observations))
    
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
    # action_scale = config['env']['action_max']
    # map_range = config['env']['map_size']
    batch_size = config['train']['batch_size']
    # buffer_size = config['train']['buffer_size']
    # data_path = config['train']['data_path']
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

    env = RadarEnv(map_config, device)

    # wandb_init(config)
    writer = SummaryWriter()

    ### ALGO Logic: Storage setup
    num_steps = config['train']['max_episode_length']
    num_envs = 1
    # obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    obs = TensorDict(
        {
            "heat_map": MemoryMappedTensor.empty(
                (num_steps, *observation_img_size),
                dtype=torch.float32,
            ),
            "goal_direction": MemoryMappedTensor.empty((num_steps, 2), dtype=torch.float32)
        },
            batch_size=[num_steps],
            device=device,
        )
    actions = torch.zeros((num_steps, num_envs) + action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)


    ### TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = env.reset()
    next_done = torch.zeros(1).to(device)

    batch_size = config['train']['batch_size']
    minibatch_size = config['train']['minibatch_size']
    num_iterations = config['train']['max_training_iterations']
    critic_warmup_iterations = config['train']['critic_warmup_iterations']

    gamma = config['train']['gamma']
    gae_lambda = config['train']['gae_lambda']
    update_epochs = config['train']['update_epochs']
    norm_adv = config['train']['norm_adv']
    clip_coef = config['train']['clip_coef']
    clip_vloss = config['train']['clip_vloss']
    ent_coef = config['train']['ent_coef']
    vf_coef = config['train']['vf_coef']
    max_grad_norm = config['train']['max_grad_norm']
    target_kl = config['train']['target_kl']

    ### Set up neural network modules.
    pretrained_model_path = config['train']['load_model']
    model_dict = torch.load(pretrained_model_path)
    actor = ActorNet(observation_space, action_space, config['train']['policy_specs']['actor_net_hidden'], hidden_act=nn.ReLU).to(device)
    actor.load_state_dict(model_dict['actor'])
    print("Load pretrained actor from: ", pretrained_model_path)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad = False

    critic = CriticNet(observation_space).to(device)
    try:
        pretrained_critic_model_path = config['train']['load_critic_model']
        critic_model_dict = torch.load(pretrained_critic_model_path)
        critic.load_state_dict(critic_model_dict)
        print("Load pretrained critic from: ", pretrained_critic_model_path)
    except:
        pass

    actor_lr = config['train']['actor_lr']
    critic_lr = config['train']['critic_lr']
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    critic_lr_schedule = CosineAnnealingLR(critic_optimizer, num_iterations)
    actor_optimizer = None
    actor_lr_schedule = None

    for iteration in range(1, num_iterations + 1):
        if iteration - 1 == critic_warmup_iterations:
            print("Start Finetuning Actor.")
            actor.train()
            for param in actor.parameters():
                param.requires_grad = True
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
            actor_lr_schedule = CosineAnnealingLR(actor_optimizer, num_iterations)
        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, prob_dist = actor(next_obs.unsqueeze(dim=0))
                value = critic(next_obs.unsqueeze(dim=0))
                values[step] = value.flatten()

            log_prob = prob_dist.distribution.log_prob(action)
            # print("Log prob: ", log_prob)
            # logprobs[step] = log_prob
            actions[step] = action
            action_idx = torch.argmax(log_prob, dim=-1, keepdim=True)
            action = torch.clamp(action_max * action, -action_max, action_max)
            action = torch.gather(action, dim=-1, index=action_idx)
            action = action.cpu().data.numpy().flatten()[0]
            # print("Log prob: ", torch.max(log_prob, dim=-1))
            logprobs[step], _ = torch.max(log_prob, dim=-1)
            # TRY NOT TO MODIFY: execute the game and log data.
            # print("Action: ", action)
            next_obs, reward, terminations, infos = env.step(action)
            # print("Termination: ", terminations)
            next_done = int(terminations == True)
            # print("Next done: ", next_done)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.tensor(next_done).to(device)
            # print("Next done: ", next_done)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = critic(next_obs.unsqueeze(dim=0))
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                _, prob_dist = actor(b_obs[mb_inds])
                entropy = prob_dist.distribution.entropy()
                newlogprob= prob_dist.distribution.log_prob(b_actions[mb_inds])
                # print("New logprob: ", newlogprob)
                newlogprob, _ = torch.max(newlogprob, dim=-1, keepdim=True)
                # print("New logprob: ", newlogprob)
                newvalue = critic(b_obs[mb_inds])
                # print("Batch log prob: ", b_logprobs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                actor_loss = pg_loss - ent_coef * entropy_loss
                critic_loss = v_loss * vf_coef

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()
                critic_lr_schedule.step()

                if iteration >= critic_warmup_iterations:
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    actor_optimizer.step()
                    actor_lr_schedule.step()


            # if target_kl is not None and approx_kl > target_kl:
            #     break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ### Log with wandb
        # log_dict = {}
        # if iteration >= critic_warmup_iterations:
        #     log_dict['actor_learning_rate'] = actor_optimizer.param_groups[0]["lr"]
        # log_dict['critic_learning_rate'] = critic_optimizer.param_groups[0]["lr"]
        # log_dict['value_loss'] = v_loss.item()
        # log_dict['policy_loss'] = pg_loss.item()
        # log_dict['entropy'] = entropy_loss.item()
        # log_dict['old_approx_kl'] = old_approx_kl.item()
        # log_dict['approx_kl'] =  approx_kl.item()
        # log_dict['clipfrac'] = np.mean(clipfracs)
        # log_dict['explained_variance'] = explained_var
        # wandb.log(log_dict, step=iteration)


        ### Log with tensorboard
        if iteration % 500 == 0:
            print("Logging at iteration: ", iteration)
            writer.add_scalar('Loss/value_loss', v_loss.item(), iteration)
            writer.add_scalar('Loss/policy_loss', pg_loss.item(), iteration)
            writer.add_scalar('Loss/entropy', entropy_loss.item(), iteration)
            if iteration >= critic_warmup_iterations:
                writer.add_scalar('lr/actor_learning_rate', actor_optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar('lr/critic_learning_rate',  critic_optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar('old_approx_kl',  old_approx_kl.item(), iteration)
            writer.add_scalar('approx_kl',  approx_kl.item(), iteration)
            writer.add_scalar('clipfrac',  np.mean(clipfracs), iteration)
            writer.add_scalar('explained_variance',  explained_var, iteration)



        if iteration % 10_000 == 0:
            print("Iteration: ", iteration)
            torch.save(
                critic.state_dict(),
                os.path.join(os.path.join(checkpoint_path, run_name), f"critic_checkpoint_{iteration}.pt"),
            )
            if iteration >= critic_warmup_iterations:
                torch.save(
                    actor.state_dict(),
                    os.path.join(os.path.join(checkpoint_path, run_name), f"actor_checkpoint_{iteration}.pt"),
                )
def get_args():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Finetune a pretrained policy with PPO.'
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