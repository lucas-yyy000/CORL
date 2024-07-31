import torch
import torch.nn as nn
from typing import List, Dict
import gymnasium as gym
from torch.distributions import Normal

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

def make_proba_distribution(action_dim, dist_kwargs):
    if dist_kwargs is None:
        dist_kwargs = {}

    return DiagGaussianDistribution(action_dim, **dist_kwargs)



class DiagGaussianDistribution(torch.distributions.Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions =  nn.Sequential(nn.Linear(latent_dim, self.action_dim), nn.Tanh())
        # TODO: allow action dependent std
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self):
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()
    
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
class ActorNet(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes,
                 hidden_act=nn.ReLU):
        super().__init__()
        if not isinstance(hidden_sizes, list):
            raise TypeError('hidden_sizes should be a list')
        self.action_space = action_space
        action_dim = action_space.shape[0]
        self.feature_extractor = FeatureExtractor(observation_space)
        in_size = self.feature_extractor.features_dim
        mlp_extractor : List[nn.Module] = []
        for curr_layer_dim in hidden_sizes:
            mlp_extractor.append(nn.Linear(in_size, curr_layer_dim))
            mlp_extractor.append(hidden_act())
            in_size = curr_layer_dim

        self.latent_dim = in_size
        self.policy_net = nn.Sequential(*mlp_extractor)
        self.act_dist = DiagGaussianDistribution(action_dim)
        self.action_net, self.log_std = self.act_dist.proba_distribution_net(self.latent_dim)

    def forward(self, observations, deterministic=False):
        feature = self.feature_extractor(observations)
        latent = self.policy_net.forward(feature)
        mean_action = self.action_net.forward(latent)
        distribution = self.act_dist.proba_distribution(mean_action, self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape)) 

        return actions, distribution
    
class NatureCNN(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class FeatureExtractor(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False
    ):
        super().__init__()
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'heat_map':
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                if not subspace.shape:
                    extractors[key] = nn.Flatten()
                    total_concat_size += 1
                else:
                    extractors[key] = nn.Flatten()
                    # total_concat_size += gym.spaces.utils.flatdim(subspace)
                    total_concat_size += subspace.shape[0]
                # print("Key: ", key, "Subspace dimension: ", subspace.shape)

        self.extractors = nn.ModuleDict(extractors)
        self.features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            # print("Key: ", key, observations[key].shape)
            encoded_tensor_list.append(extractor(observations[key].contiguous()))
        # print("Encoder list: ", len(encoded_tensor_list))
        return torch.cat(encoded_tensor_list, dim=1)