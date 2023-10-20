"""Common aliases for type hints"""

from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np
import torch as th

from cirl_stable_baselines3.common import callbacks
from cirl_stable_baselines3.common.vec_env import VecEnv

GymEnv = Union[gym.Env, VecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
TensorDict = Dict[str, th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RolloutBufferWithCostSamples(NamedTuple):
    orig_observations: th.Tensor
    observations: th.Tensor
    actions: th.Tensor
    old_log_prob: th.Tensor
    old_reward_values: th.Tensor
    reward_advantages: th.Tensor
    reward_returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor


class RolloutBufferWithCostCodeSamples(NamedTuple):
    orig_observations: th.Tensor
    observations: th.Tensor
    actions: th.Tensor
    codes: th.Tensor
    pos_latent_signals: th.Tensor
    neg_latent_signals: th.Tensor
    old_reward_values: th.Tensor
    reward_advantages: th.Tensor
    reward_returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor
    diversity_values: th.Tensor
    diversity_advantages: th.Tensor
    diversity_returns: th.Tensor
    old_log_prob: th.Tensor


class LagrangianBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool
