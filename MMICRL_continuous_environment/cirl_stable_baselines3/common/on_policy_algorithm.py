import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch as th

from cirl_stable_baselines3.common import logger, utils
from cirl_stable_baselines3.common.base_class import BaseAlgorithm
from cirl_stable_baselines3.common.buffers import (RolloutBuffer,
                                                   CustomRolloutBuffer,
                                                   RolloutBufferWithCost, RolloutBufferWithCostCode)
from cirl_stable_baselines3.common.callbacks import BaseCallback
from cirl_stable_baselines3.common.policies import ActorCriticPolicy
from cirl_stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from cirl_stable_baselines3.common.utils import safe_mean
from cirl_stable_baselines3.common.vec_env import VecEnv, VecNormalize, VecNormalizeWithCost

# from cirl_stable_baselines3.common.dual_variable import DualVariable
from utils.model_utils import build_code, diversity_return_function


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable],
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = CustomRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            new_original_obs = env.get_original_obs() if isinstance(env, VecNormalize) else new_obs

            # if infos[0]['is_off_road']:
            #     print('debugging')

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, self._last_original_obs, new_obs, new_original_obs,
                               actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_orginal_obs = new_original_obs
            self._last_dones = dones
            # if dones[0]:
            #     print('debug')

        self.extras = {'last_values': values, 'dones': dones}
        rollout_buffer.compute_returns_and_advantage(values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  safe_mean([ep_info["reward"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["len"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                # logger.dump(step=self.num_timesteps)
            tmp = logger.Logger.CURRENT
            tmp_dict = tmp.name_to_value
            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


class OnPolicyWithCostAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable],
            n_steps: int,
            reward_gamma: float,
            reward_gae_lambda: float,
            cost_gamma: float,
            cost_gae_lambda: float,
            ent_coef: float,
            reward_vf_coef: float,
            cost_vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(OnPolicyWithCostAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.ent_coef = ent_coef
        self.reward_vf_coef = reward_vf_coef
        self.cost_vf_coef = cost_vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBufferWithCost(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            reward_gamma=self.reward_gamma,
            reward_gae_lambda=self.reward_gae_lambda,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
            n_envs=self.n_envs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBufferWithCost, n_rollout_steps: int,
            cost_function: Union[str, Callable]
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: (RolloutBufferWithCost) Buffer to fill with rollouts
        :param n_steps: (int) Number of experiences to collect per environment
        :param cost_function: (str, Callable) Either a callable that returns the cost
            of a state-action marginal, or the key in the info dict corresponding to
            the cost
        :return: (bool) True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, reward_values, cost_values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            orig_obs = env.get_original_obs() if isinstance(env, VecNormalize) else new_obs
            if type(cost_function) is str:
                # Need to get cost from environment.
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(env, VecNormalizeWithCost):
                    orig_costs = env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(orig_obs.copy(), clipped_actions)
                orig_costs = costs

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, self._last_original_obs, new_obs, orig_obs, actions,
                               rewards, costs, orig_costs, self._last_dones, reward_values,
                               cost_values, log_probs)
            self._last_obs = new_obs
            self._last_original_obs = orig_obs
            self._last_dones = dones

        rollout_buffer.compute_returns_and_advantage(reward_values, cost_values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            cost_function: Union[str, Callable],
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyWithCostAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyWithCostAlgorithm":

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        def training_infos(itr):
            fps = int(self.num_timesteps / (time.time() - self.start_time))
            logger.record("time/iterations", itr, exclude="tensorboard")
            logger.record("time/fps", fps)
            logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
            logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            # if self.info_buffers is not None and self.verbose == 2:
            #     for key in self.info_buffers:
            #         if len(self.info_buffers[key]) > 0:
            #             logger.record("infos/" + key, safe_mean(self.info_buffers[key]))
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                keywords = {key for ep_info in self.ep_info_buffer for key in ep_info.keys()}
                keywords -= {'reward', 'len', 'time'}
                for keyword in keywords:
                    if keyword == 'env':
                        continue  # we can not average the env id.
                    else:
                        logger.record(f"rollout/ep_{keyword}_mean",
                                      safe_mean([ep_info[keyword] for ep_info in self.ep_info_buffer]))
                        logger.record(f"rollout/ep_{keyword}_max",
                                      np.max([ep_info[keyword] for ep_info in self.ep_info_buffer]))
                        logger.record(f"rollout/ep_{keyword}_min",
                                      np.min([ep_info[keyword] for ep_info in self.ep_info_buffer]))
                logger.record("rollout/ep_rew_mean", safe_mean([ep_info["reward"] for ep_info in self.ep_info_buffer]))
                logger.record("rollout/ep_len_mean", safe_mean([ep_info["len"] for ep_info in self.ep_info_buffer]))

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.n_steps,
                                                      cost_function)
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                training_infos(iteration)
                # logger.dump(step=self.num_timesteps)

            self.train()

        training_infos(iteration + 1)
        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


class OnPolicyWithCostAndCodeAlgorithm(BaseAlgorithm):
    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable],
            n_steps: int,
            reward_gamma: float,
            reward_gae_lambda: float,
            cost_gamma: float,
            cost_gae_lambda: float,
            diversity_gamma: float,
            diversity_gae_lambda: float,
            ent_coef: float,
            reward_vf_coef: float,
            cost_vf_coef: float,
            diversity_vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            latent_dim: int,
            n_probings: int,
            contrastive_weight: float,
            cid: int,
            log_cost: bool = True,
            contrastive_augment_type: bool = False,
            loss_type: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(OnPolicyWithCostAndCodeAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.reward_gamma = reward_gamma
        self.reward_gae_lambda = reward_gae_lambda
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.diversity_gamma = diversity_gamma
        self.diversity_gae_lambda = diversity_gae_lambda
        self.ent_coef = ent_coef
        self.reward_vf_coef = reward_vf_coef
        self.cost_vf_coef = cost_vf_coef
        self.diversity_vf_coef = diversity_vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.code_dim = latent_dim
        self.code_id = cid
        self.n_probings = n_probings
        self.contrastive_weight = contrastive_weight
        self.log_cost = log_cost
        self.contrastive_augment_type = contrastive_augment_type
        self.loss_type = loss_type

        if _init_setup_model:
            self._setup_model()

        if isinstance(self.env, VecNormalize):
            self._last_latent_codes = build_code(code_axis=[self.code_id for _ in range(self.env.venv.num_envs)],
                                                 code_dim=self.env.venv.latent_dim,
                                                 num_envs=self.env.venv.num_envs)
        else:
            self._last_latent_codes = build_code(code_axis=[self.code_id for _ in range(self.env.num_envs)],
                                                 code_dim=self.env.latent_dim,
                                                 num_envs=self.env.num_envs)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBufferWithCostCode(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            code_dim=self.code_dim,
            reward_gamma=self.reward_gamma,
            reward_gae_lambda=self.reward_gae_lambda,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
            diversity_gamma=self.diversity_gamma,
            diversity_gae_lambda=self.diversity_gae_lambda,
            n_envs=self.n_envs,
            n_probings=self.n_probings
        )

        self.policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBufferWithCostCode,
            n_rollout_steps: int, cost_info_str: str, latent_info_str: str, density_loss: float
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: (RolloutBufferWithCost) Buffer to fill with rollouts
        :param n_rollout_steps: (int) Number of experiences to collect per environment
        :param cost_info_str: the key in the info dict corresponding to the cost
        :param latent_info_str: the key in the info dict corresponding to the latent code
        :return: (bool) True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                codes_tensor = th.as_tensor(self._last_latent_codes, dtype=torch.float64).to(self.device)
                input_tensor = th.cat([obs_tensor, codes_tensor], dim=1)
                if self.contrastive_augment_type == 'calculate advantages':  # directly augment contrastive loss to rewards
                    actions, reward_values, cost_values, diversity_values, log_probs = self.policy.forward(input_tensor)
                else:
                    actions, reward_values, cost_values, log_probs = self.policy.forward(input_tensor)
                    diversity_values = torch.zeros_like(cost_values)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            # we overload the action parameter a little for the convenience of pass latent code
            # clipped_actions_codes = np.concatenate([clipped_actions, self._last_latent_codes], axis=1)
            new_obs, rewards, dones, infos = env.step_with_code(clipped_actions, self._last_latent_codes)
            orig_obs = env.get_original_obs() if isinstance(env, VecNormalize) else new_obs
            if type(cost_info_str) is str:
                # Need to get cost from environment.
                costs_signals = np.array([info[cost_info_str] for info in infos])
                new_latent_codes = np.array([info['new_code'] for info in infos])
                pos_latent_signals = np.array([info[latent_info_str]['pos'] for info in infos])
                neg_latent_signals = np.array([info[latent_info_str]['neg'] for info in infos])
                if isinstance(env, VecNormalizeWithCost):
                    orig_costs = env.get_original_cost()
                else:
                    orig_costs = costs_signals
                # apply logarithm
                # if self.log_cost:
                #     costs_signals = np.log(costs_signals + 1e-5)  # normalized costs, pls do not use this one
                #     orig_costs = np.log(orig_costs + 1e-5)  # unnormalized costs, this is more reasonable
            else:
                raise ValueError("This part is not yet done.")
                # costs = cost_function(orig_obs.copy(), clipped_actions)
                # orig_costs = costs
            diversity_score = np.zeros_like(rewards)  # dummy score
            contrastive_loss = 0
            if self.contrastive_augment_type == 'calculate advantages' \
                    or self.contrastive_augment_type == 'reward augmentation':
                if self.loss_type == 'probing_vectors':
                    contrastive_loss = diversity_return_function(observations=self._last_original_obs,
                                                                 actions=actions,
                                                                 pos_latent_signals=pos_latent_signals,
                                                                 neg_latent_signals=neg_latent_signals)
                elif self.loss_type == 'density_loss':
                    while density_loss >= 1 or density_loss <= -1:
                        density_loss /= 10
                    density_loss = np.array(density_loss)
                    contrastive_loss = density_loss

                if self.contrastive_augment_type == 'reward augmentation':  # directly augment contrastive loss to rewards
                    rewards = rewards - self.contrastive_weight * contrastive_loss
                if self.contrastive_augment_type == 'calculate advantages':
                    diversity_score = contrastive_loss
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(obs=self._last_obs,
                               orig_obs=self._last_original_obs,
                               new_obs=new_obs,
                               new_orig_obs=orig_obs,
                               action=actions,
                               code=self._last_latent_codes,
                               pos_posterior_signal=pos_latent_signals,
                               neg_posterior_signal=neg_latent_signals,
                               cost=costs_signals,
                               orig_cost=orig_costs,
                               cost_value=cost_values,
                               reward=rewards,
                               reward_value=reward_values,
                               diversity_score=diversity_score,
                               diversity_value=diversity_values,
                               done=self._last_dones,
                               log_prob=log_probs)
            self._last_obs = new_obs
            self._last_original_obs = orig_obs
            self._last_dones = dones
            self._last_latent_codes = new_latent_codes
            # print(self._last_latent_codes)

        rollout_buffer.compute_returns_and_advantage(reward_values, cost_values, diversity_values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            cost_info_str: str,
            latent_info_str: str,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyWithCostCodeAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            density_loss: float = 0.0,
    ) -> "OnPolicyWithCostAndCodeAlgorithm":

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        def training_infos(itr):
            fps = int(self.num_timesteps / (time.time() - self.start_time))
            logger.record("time/iterations", itr, exclude="tensorboard")
            logger.record("time/fps", fps)
            logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
            logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

            # if self.info_buffers is not None and self.verbose == 2:
            #     for key in self.info_buffers:
            #         if len(self.info_buffers[key]) > 0:
            #             logger.record("infos/" + key, safe_mean(self.info_buffers[key]))
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                keywords = {key for ep_info in self.ep_info_buffer for key in ep_info.keys()}
                keywords -= {'reward', 'len', 'time'}
                for keyword in keywords:
                    if keyword == 'env':
                        continue  # we can not average the env id.
                    else:
                        logger.record(f"rollout/ep_{keyword}_mean",
                                      safe_mean([ep_info[keyword] for ep_info in self.ep_info_buffer]))
                        logger.record(f"rollout/ep_{keyword}_max",
                                      np.max([ep_info[keyword] for ep_info in self.ep_info_buffer]))
                        logger.record(f"rollout/ep_{keyword}_min",
                                      np.min([ep_info[keyword] for ep_info in self.ep_info_buffer]))
                logger.record("rollout/ep_rew_mean", safe_mean([ep_info["reward"] for ep_info in self.ep_info_buffer]))
                logger.record("rollout/ep_len_mean", safe_mean([ep_info["len"] for ep_info in self.ep_info_buffer]))

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(env=self.env,
                                                      callback=callback,
                                                      rollout_buffer=self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps,
                                                      cost_info_str=cost_info_str,
                                                      latent_info_str=latent_info_str,
                                                      density_loss=density_loss, )
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                training_infos(iteration)
                # logger.dump(step=self.num_timesteps)

            self.train()

        training_infos(iteration + 1)
        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
