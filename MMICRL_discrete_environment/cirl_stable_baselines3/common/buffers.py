import warnings
from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from cirl_stable_baselines3.common.preprocessing import (get_action_dim,
                                                         get_obs_shape)
from cirl_stable_baselines3.common.type_aliases import (
    ReplayBufferSamples, RolloutBufferSamples, RolloutBufferWithCostSamples, RolloutBufferWithCostCodeSamples)
from cirl_stable_baselines3.common.vec_env import VecNormalize


class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(obs: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_obs(obs).astype(np.float32)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_value: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_value:
        :param dones:

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: th.Tensor, log_prob: th.Tensor
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

# ==========================================================================
# Custom Rollout Buffer
# Extends the Rollout Buffer To Store
# (1) Unnormalized original observation
# (2) Nromalized next observation
# (3) Unnormalized next observation
# ==========================================================================

class CustomRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(CustomRolloutBuffer, self).__init__(buffer_size=buffer_size, observation_space=observation_space,
                                            action_space=action_space, device=device, gae_lambda=gae_lambda,
                                            gamma=gamma, n_envs=n_envs)
    def reset(self) -> None:
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.orig_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_orig_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        super(CustomRolloutBuffer, self).reset()

    def add(
            self, obs: np.ndarray, orig_obs:np.ndarray, new_obs: np.ndarray, new_orig_obs: np.ndarray,
            action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: th.Tensor, log_prob: th.Tensor
    ) -> None:
        """
        :param obs: Observation
        :param orig_obs: Original observation
        :param new_obs: Next observation (to which agent transitions)
        """
        self.orig_observations[self.pos] = np.array(orig_obs).copy()
        self.new_observations[self.pos] = np.array(new_obs).copy()
        self.new_orig_observations[self.pos] = np.array(new_orig_obs).copy()
        super(CustomRolloutBuffer, self).add(obs=obs, action=action, reward=reward,done=done,
                                             value=value, log_prob=log_prob)

# ==========================================================================
# Rollout Buffer With Cost
# Extends the Rollout Buffer To Store
# (1) Unnormalized original observation
# (2) Nromalized next observation
# (3) Unnormalized next observation
# (4) Cost
# (5) Orig Cost (Unnormalized cost)
# (6) Cost Returns
# (7) Cost Advantages
# ==========================================================================


class RolloutBufferWithCost(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        reward_gamma: float = 0.99,
        reward_gae_lambda: float = 1,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 1,
        n_envs: int = 1,
    ):

        super(RolloutBufferWithCost, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.observations, self.orig_observations, self.actions, self.rewards, self.advantages = None, None, None, None, None
        self.new_observations, self.new_orig_observations = None, None
        self.reward_returns, self.cost_returns, self.dones, self.values, self.log_probs = None, None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.orig_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_orig_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Rewards
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Costs
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.orig_costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready = False
        super(RolloutBufferWithCost, self).reset()

    def _compute_returns_and_advantage(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
            gamma: float,
            gae_lambda: float,
            last_value: th.Tensor,
            last_dones: np.ndarray,
            advantages: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param rewards:
        :param values:
        :param dones:
        :param gamma:
        :param gae_lambda:
        :param last_value:
        :param last_dones:
        :param advantages
        :return advantages
        :return returns

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value = values[step + 1]
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        returns = advantages + values

        return returns, advantages

    def compute_returns_and_advantage(self, reward_last_value: th.Tensor, cost_last_value: th.Tensor,
                                      dones: np.ndarray) -> None:
        self.reward_returns, self.reward_advantages = self._compute_returns_and_advantage(
                self.rewards, self.reward_values, self.dones, self.reward_gamma, self.reward_gae_lambda,
                reward_last_value, dones, self.reward_advantages
        )
        self.cost_returns, self.cost_advantages = self._compute_returns_and_advantage(
                self.costs, self.cost_values, self.dones, self.cost_gamma, self.cost_gae_lambda,
                cost_last_value, dones, self.cost_advantages
        )

    def add(self, obs: np.ndarray, orig_obs: np.ndarray, new_obs: np.ndarray, new_orig_obs: np.ndarray,
            action: np.ndarray, reward: np.ndarray, cost: np.ndarray, orig_cost: np.ndarray,
            done: np.ndarray, reward_value: th.Tensor, cost_value: th.Tensor,
            log_prob: th.Tensor) -> None:
        """
        :param obs: Observation
        :param orig_obs: Original observation
        :param new_obs: Next observation (to which agent transitions)
        :param action: Action
        :param reward:
        :param cost:
        :param orig_cost: original cost
        :param done: End of episode signal.
        :param reward_value: estimated reward value of the current state
            following the current policy.
        :param cost_value: estimated cost value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.orig_observations[self.pos] = np.array(orig_obs).copy()
        self.new_observations[self.pos] = np.array(new_obs).copy()
        self.new_orig_observations[self.pos] = np.array(new_orig_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.reward_values[self.pos] = reward_value.clone().cpu().numpy().flatten()
        self.costs[self.pos] = np.array(cost).copy()
        self.orig_costs[self.pos] = np.array(orig_cost).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["orig_observations", "observations", "actions", "log_probs", "reward_values",
                           "reward_advantages", "reward_returns", "cost_values", "cost_advantages",
                           "cost_returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.orig_observations[batch_inds],
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.log_probs[batch_inds].flatten(),
            self.reward_values[batch_inds].flatten(),
            self.reward_advantages[batch_inds].flatten(),
            self.reward_returns[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
        )
        return RolloutBufferWithCostSamples(*tuple(map(self.to_torch, data)))


# ==========================================================================
# Rollout Buffer With Cost
# Extends the Rollout Buffer To Store
# (1) Unnormalized original observation
# (2) Nromalized next observation
# (3) Unnormalized next observation
# (4) latent code
# (5) Cost
# (6) Orig Cost (Unnormalized cost)
# (7) Cost Returns
# (8) Cost Advantages
# ==========================================================================


class RolloutBufferWithCostCode(BaseBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            code_dim: int,
            device: Union[th.device, str] = "cpu",
            reward_gamma: float = 0.99,
            reward_gae_lambda: float = 1,
            cost_gamma: float = 0.99,
            cost_gae_lambda: float = 1,
            diversity_gamma: float = 0.99,
            diversity_gae_lambda: float = 1,
            n_envs: int = 1,
            n_probings: int = 1,
    ):

        super(RolloutBufferWithCostCode, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.latent_dim = code_dim
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.diversity_gamma = diversity_gamma
        self.diversity_gae_lambda = diversity_gae_lambda
        self.observations, self.orig_observations, self.actions, self.rewards, self.advantages = \
            None, None, None, None, None
        self.pos_latent_signals, self.neg_latent_signals, self.codes = None, None, None
        self.new_observations, self.new_orig_observations = None, None
        self.reward_returns, self.cost_returns, self.dones, self.values, self.log_probs = None, None, None, None, None
        self.generator_ready = False
        self.n_probings = n_probings
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.orig_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_orig_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.codes = np.zeros((self.buffer_size, self.n_envs, self.latent_dim), dtype=np.float32)
        self.pos_latent_signals = np.zeros((self.buffer_size,
                                            self.n_envs,
                                            self.n_probings,
                                            self.action_dim + self.obs_shape[0]), dtype=np.float32)
        self.neg_latent_signals = np.zeros((self.buffer_size,
                                            self.n_envs,
                                            self.latent_dim - 1,
                                            self.n_probings,
                                            self.action_dim + self.obs_shape[0]), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Rewards
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reward_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Costs
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.orig_costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Costs
        self.diversities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.diversity_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.diversity_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.diversity_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready = False
        super(RolloutBufferWithCostCode, self).reset()

    def _compute_returns_and_advantage(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
            gamma: float,
            gae_lambda: float,
            last_value: th.Tensor,
            last_dones: np.ndarray,
            advantages: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param rewards:
        :param values:
        :param dones:
        :param gamma:
        :param gae_lambda:
        :param last_value:
        :param last_dones:
        :param advantages
        :return advantages
        :return returns

        """
        # convert to numpy
        last_value = last_value.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value = values[step + 1]
            delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        returns = advantages + values

        return returns, advantages

    def compute_returns_and_advantage(self, reward_last_value: th.Tensor, cost_last_value: th.Tensor,
                                      diversity_last_value: th.Tensor, dones: np.ndarray) -> None:
        self.reward_returns, self.reward_advantages = self._compute_returns_and_advantage(
                self.rewards, self.reward_values, self.dones, self.reward_gamma, self.reward_gae_lambda,
                reward_last_value, dones, self.reward_advantages
        )
        self.cost_returns, self.cost_advantages = self._compute_returns_and_advantage(
                self.costs, self.cost_values, self.dones, self.cost_gamma, self.cost_gae_lambda,
                cost_last_value, dones, self.cost_advantages
        )
        self.diversity_returns, self.diversity_advantages = self._compute_returns_and_advantage(
                self.diversities, self.diversity_values, self.dones, self.diversity_gamma, self.diversity_gae_lambda,
                diversity_last_value, dones, self.diversity_advantages
        )

    def add(self, obs: np.ndarray, orig_obs: np.ndarray, new_obs: np.ndarray, new_orig_obs: np.ndarray,
            action: np.ndarray, code: np.ndarray, pos_posterior_signal: np.ndarray, neg_posterior_signal: np.ndarray,
            reward: np.ndarray, reward_value: th.Tensor, cost: np.ndarray, orig_cost: np.ndarray, cost_value: th.Tensor,
            diversity_score: np.ndarray, diversity_value: th.Tensor,
            done: np.ndarray, log_prob: th.Tensor) -> None:
        """
        :param obs: Observation
        :param orig_obs: Original observation
        :param new_obs: Next observation (to which agent transitions)
        :param new_orig_obs: Next Original observation (to which agent transitions)
        :param action: Action
        :param reward:
        :param code: latent code
        :param pos_posterior_signal: pos_posterior_signal
        :param neg_posterior_signal: pos_posterior_signal
        :param cost:
        :param orig_cost: original cost
        :param done: End of episode signal.
        :param reward_value: estimated reward value of the current state
            following the current policy.
        :param cost_value: estimated cost value of the current state
            following the current policy.
        :param diversity_score: diversity_score
        :param diversity_score: diversity_score
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs).copy()
        self.orig_observations[self.pos] = np.array(orig_obs).copy()
        self.new_observations[self.pos] = np.array(new_obs).copy()
        self.new_orig_observations[self.pos] = np.array(new_orig_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.codes[self.pos] = np.array(code).copy()
        self.pos_latent_signals[self.pos] = np.array(pos_posterior_signal).copy()
        self.neg_latent_signals[self.pos] = np.array(neg_posterior_signal).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.reward_values[self.pos] = reward_value.clone().cpu().numpy().flatten()
        self.costs[self.pos] = np.array(cost).copy()
        self.orig_costs[self.pos] = np.array(orig_cost).copy()
        self.cost_values[self.pos] = cost_value.clone().cpu().numpy().flatten()
        self.diversities[self.pos] = np.array(diversity_score).copy()
        self.diversity_values[self.pos] = diversity_value.clone().cpu().numpy().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in ["orig_observations", "observations", "actions",
                           "codes", "pos_latent_signals", "neg_latent_signals",
                           "reward_values", "reward_advantages", "reward_returns",
                           "cost_values", "cost_advantages", "cost_returns",
                           "diversity_values", "diversity_advantages", "diversity_returns",
                           "log_probs"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferWithCostCodeSamples:
        data = (
            self.orig_observations[batch_inds],
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.codes[batch_inds],
            self.pos_latent_signals[batch_inds],
            self.neg_latent_signals[batch_inds],
            self.reward_values[batch_inds].flatten(),
            self.reward_advantages[batch_inds].flatten(),
            self.reward_returns[batch_inds].flatten(),
            self.cost_values[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.diversity_values[batch_inds].flatten(),
            self.diversity_advantages[batch_inds].flatten(),
            self.diversity_returns[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
        )
        return RolloutBufferWithCostCodeSamples(*tuple(map(self.to_torch, data)))
