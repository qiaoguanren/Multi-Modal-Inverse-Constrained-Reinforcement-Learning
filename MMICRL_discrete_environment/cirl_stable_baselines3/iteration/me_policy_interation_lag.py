import copy
import os
from abc import ABC
from typing import Any, Callable, Dict, Optional, Type, Union

import random
import numpy as np
import torch
from tqdm import tqdm

from cirl_stable_baselines3.common.dual_variable import DualVariable
from cirl_stable_baselines3.common.type_aliases import GymEnv
from cirl_stable_baselines3.common import logger
from cirl_stable_baselines3.common.vec_env import VecNormalizeWithCost, VecNormalize
from cirl_stable_baselines3.iteration import PolicyIterationLagrange
from utils.model_utils import to_np, build_code
from cirl_stable_baselines3.common.utils import set_random_seed
from utils.model_utils import diversity_return_function


class MixtureExpertPolicyIterationLagrange(PolicyIterationLagrange):

    def __init__(self,
                 env: Union[GymEnv, str],
                 max_iter: int,
                 n_actions: int,
                 height: int,  # table length
                 width: int,  # table width
                 terminal_states: int,
                 stopping_threshold: float,
                 contrastive_weight: float,
                 seed: int,
                 aid: int,
                 latent_dim: int,
                 gamma: float = 0.99,
                 v0: float = 0.0,
                 budget: float = 0.,
                 penalty_initial_value: float = 1,
                 penalty_learning_rate: float = 0.01,
                 penalty_min_value: Optional[float] = None,
                 penalty_max_value: Optional[float] = None,
                 log_file=None,
                 loss_type: bool = False,
                 device: Union[torch.device, str] = "auto",
                 ):
        self.agent_id = aid
        self.contrastive_weight = contrastive_weight
        self.latent_dim = latent_dim
        self.device = device
        self.loss_type = loss_type
        super(MixtureExpertPolicyIterationLagrange, self).__init__(
            env=env,
            max_iter=max_iter,
            n_actions=n_actions,
            height=height,
            width=width,
            terminal_states=terminal_states,
            stopping_threshold=stopping_threshold,
            seed=seed,
            gamma=gamma,
            v0=v0,
            budget=budget,
            penalty_initial_value=penalty_initial_value,
            penalty_learning_rate=penalty_learning_rate,
            penalty_min_value=penalty_min_value,
            penalty_max_value=penalty_max_value,
            log_file=log_file,
        )

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=True)
        if self.env is not None:
            self.env.seed(seed)

    def _setup_model(self) -> None:
        self.dual = DualVariable(self.budget,
                                 self.penalty_learning_rate,
                                 self.penalty_initial_value,
                                 min_clamp=self.penalty_min_value,
                                 max_clamp=self.penalty_max_value,
                                 )
        self.set_random_seed(self.seed)
        self.v_m = self.get_init_v()
        self.pi = self.get_equiprobable_policy()

        if isinstance(self.env, VecNormalize):
            self._last_latent_codes = build_code(code_axis=[self.agent_id for _ in range(self.env.venv.num_envs)],
                                                 code_dim=self.env.venv.latent_dim,
                                                 num_envs=self.env.venv.num_envs)
        else:
            self._last_latent_codes = build_code(code_axis=[self.agent_id for _ in range(self.env.num_envs)],
                                                 code_dim=self.env.latent_dim,
                                                 num_envs=self.env.num_envs)

    def predict(self, obs, state, deterministic=True):
        if obs.shape[1] == self.latent_dim + 2:
            obs = obs[:, :-self.latent_dim]
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
        action = random.choice(best_actions)
        return np.asarray([action]), state

    def step(self, action):
        return self.env.step_with_code(actions=np.asarray([action]),
                                       codes=self._last_latent_codes
                                       )

    def bellman_update(self, old_v, x, y, cost_function, density_loss,latent_info_str):
        if [x, y] in self.terminal_states:
            return
        total = 0
        for action in range(self.n_actions):
            states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
            assert states[0][0] == x and states[0][1] == y
            # Get next state
            s_primes, rewards, dones, infos = self.step(action)
            # Get cost from environment.
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
                new_latent_codes = np.array([info['new_code'] for info in infos])
            else:
                raise ValueError("This part is not yet done.")
            gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
            current_penalty = self.dual.nu().item()
            lag_costs = current_penalty * orig_costs[0]
            if self.loss_type == "density_loss":
                total += self.pi[x, y, action] * (rewards[0] - lag_costs + gamma_values + self.contrastive_weight * density_loss)
            elif self.loss_type == "probing_vectors" or self.loss_type == "":
                total += self.pi[x, y, action] * (rewards[0] - lag_costs + gamma_values)
            self._last_latent_codes = new_latent_codes
        self.v_m[x, y] = total

    def save(self, save_path):
        state_dict = dict(
            pi=self.pi,
            v_m=self.v_m,
            gamma=self.gamma,
            max_iter=self.max_iter,
            n_actions=self.n_actions,
            terminal_states=self.terminal_states,
            seed=self.seed,
            height=self.height,
            width=self.width,
            budget=self.budget,
            num_timesteps=self.num_timesteps,
            stopping_threshold=self.stopping_threshold,
            latent_dim=self.latent_dim,
            aid=self.agent_id,
        )
        torch.save(state_dict, save_path)

    def policy_improvement(self, cost_function,density_loss,latent_info_str):
        """Applies the Policy Improvement step."""
        policy_stable = True

        # Iterate states
        for x in range(self.height):
            for y in range(self.width):
                if [x, y] in self.terminal_states:
                    continue
                old_pi = self.pi[x, y, :].copy()

                # Iterate all actions
                action_values = []
                for action in range(self.n_actions):
                    states = self.env.reset_with_values(info_dicts=[{'states': [x, y]}])
                    assert states[0][0] == x and states[0][1] == y
                    # Compute next state
                    s_primes, rewards, dones, infos = self.step(action=action)
                    # Get cost from environment.
                    if type(cost_function) is str:
                        costs = np.array([info.get(cost_function, 0) for info in infos])
                        if isinstance(self.env, VecNormalizeWithCost):
                            orig_costs = self.env.get_original_cost()
                        else:
                            orig_costs = costs
                        new_latent_codes = np.array([info['new_code'] for info in infos])
                    else:
                        raise ValueError("This part is not yet done.")
                    current_penalty = self.dual.nu().item()
                    lag_costs = current_penalty * orig_costs[0]
                    # Get value
                    curr_val = 0.0
                    if self.loss_type == "density_loss":
                        curr_val = rewards[0] - lag_costs + self.gamma * self.v_m[s_primes[0][0], s_primes[0][1]] + self.contrastive_weight*density_loss
                    elif self.loss_type == "probing_vectors" or self.loss_type == "":
                        curr_val = rewards[0] - lag_costs + self.gamma * self.v_m[s_primes[0][0], s_primes[0][1]]
                    action_values.append(curr_val)
                    self._last_latent_codes = new_latent_codes
                best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()
                # Define new policy
                self.define_new_policy(x, y, best_actions)

                # Check whether the policy has changed
                if not (old_pi == self.pi[x, y, :]).all():
                    policy_stable = False

        return policy_stable


def load_pi(model_path, iter_msg, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)

    state_dict = torch.load(model_path)

    pi = state_dict["pi"]
    v_m = state_dict["v_m"]
    gamma = state_dict["gamma"]
    max_iter = state_dict["max_iter"]
    n_actions = state_dict["n_actions"]
    terminal_states = state_dict["terminal_states"]
    seed = state_dict["seed"]
    height = state_dict["height"]
    width = state_dict["width"]
    budget = state_dict["budget"]
    stopping_threshold = state_dict["stopping_threshold"]

    create_iteration_agent = lambda: PolicyIterationLagrange(
        env=None,
        max_iter=max_iter,
        n_actions=n_actions,
        height=height,  # table length
        width=width,  # table width
        terminal_states=terminal_states,
        stopping_threshold=stopping_threshold,
        seed=seed,
        gamma=gamma,
        budget=budget, )
    iteration_agent = create_iteration_agent()
    iteration_agent.pi = pi
    iteration_agent.v_m = v_m

    return iteration_agent
