from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Any

import gym
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from cirl_stable_baselines3.common.preprocessing import (get_flattened_obs_dim,
                                                         is_image_space)
from cirl_stable_baselines3.common.utils import get_device


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space} "
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)"
        )
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


def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU, squash_output: bool = False
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.
    Modified to add support for creating cost value nets, if create_cvf is set to True.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    :param create_cvf:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
        create_cvf: bool = False,
        create_dvf: bool = False,
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        cost_value_only_layers = []  # Layer sizes of the network that only belongs to the cost value network
        diversity_value_only_layers = []  # Layer sizes of the network that only belongs to the diversity value network
        last_layer_dim_shared = feature_dim

        self.create_cvf = create_cvf
        self.create_dvf = create_dvf

        # If we also need to create a value function for cost
        if create_cvf:
            cost_value_net = []
        else:
            cost_value_net = None

        # If we also need to create a value function for diversity
        if create_dvf:
            diversity_value_net = []
        else:
            diversity_value_net = None

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]

                if create_cvf and "cvf" in layer:
                    assert isinstance(layer["cvf"], list), "Error: net_arch[-1]['cvf'] must contain a list of integers."
                    cost_value_only_layers = layer["cvf"]

                if create_dvf and "dvf" in layer:
                    assert isinstance(layer["dvf"], list), "Error: net_arch[-1]['cvf'] must contain a list of integers."
                    diversity_value_only_layers = layer["dvf"]

                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared
        if create_cvf:
            last_layer_dim_cvf = last_layer_dim_shared
        if create_dvf:
            last_layer_dim_dvf = last_layer_dim_shared
        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size, cvf_layer_size, dvf_layer_size) in enumerate(
                zip_longest(policy_only_layers, value_only_layers, cost_value_only_layers, diversity_value_only_layers
            )):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

            if cvf_layer_size is not None:  # Will be none if cost_vf is False
                assert isinstance(cvf_layer_size, int), "Error: net_arch[-1]['cvf'] must only contain integers."
                cost_value_net.append(nn.Linear(last_layer_dim_cvf, cvf_layer_size))
                cost_value_net.append(activation_fn())
                last_layer_dim_cvf = cvf_layer_size

            if dvf_layer_size is not None:  # Will be none if diversity_vf is False
                assert isinstance(dvf_layer_size, int), "Error: net_arch[-1]['dvf'] must only contain integers."
                diversity_value_net.append(nn.Linear(last_layer_dim_dvf, dvf_layer_size))
                diversity_value_net.append(activation_fn())
                last_layer_dim_dvf = dvf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        if create_cvf:
            self.latent_dim_cvf = last_layer_dim_cvf
        if create_dvf:
            self.latent_dim_dvf = last_layer_dim_dvf
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        if create_cvf:
            self.cost_value_net = nn.Sequential(*cost_value_net).to(device)
        if create_dvf:
            self.diversity_value_net = nn.Sequential(*diversity_value_net).to(device)

    def forward(self, features: torch.Tensor) -> Union[
        Tuple[Any, Any, Any, Any], Tuple[Any, Any, Any], Tuple[Any, Any]]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        if self.create_cvf and self.create_dvf:
            return self.policy_net(shared_latent), self.value_net(shared_latent), \
                   self.cost_value_net(shared_latent), self.diversity_value_net(shared_latent)
        elif self.create_dvf:
            return self.policy_net(shared_latent), self.value_net(shared_latent), \
                   self.diversity_value_net(shared_latent)
        elif self.create_cvf:
            return self.policy_net(shared_latent), self.value_net(shared_latent), self.cost_value_net(shared_latent)
        else:
            return self.policy_net(shared_latent), self.value_net(shared_latent)


class ResBlock(torch.nn.Module):
    """It should be a strict resnet"""

    def __init__(self, input_dims):
        super(ResBlock, self).__init__()
        self.inputs_dims = input_dims

        dense_layer_1 = torch.nn.Linear(in_features=self.inputs_dims, out_features=self.inputs_dims)
        dense_layer_2 = torch.nn.Linear(in_features=self.inputs_dims, out_features=self.inputs_dims)

        self.model = torch.nn.Sequential(
            spectral_norm(dense_layer_1),
            torch.nn.LeakyReLU(),
            spectral_norm(dense_layer_2))
        self.latent_feature = None

    def forward(self, x):
        """
        implementing hl(x) = x+gl(x) in the paper:
        "Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness"
        https://arxiv.org/pdf/2006.10108.pdf
        refer to https://github.com/omegafragger/DDU/blob/f597744c65df4ff51615ace5e86e82ffefe1cd0f/net/resnet.py
        #
        """
        return self.model(x) + x