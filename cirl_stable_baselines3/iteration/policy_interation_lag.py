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
from cirl_stable_baselines3.common.vec_env import VecNormalizeWithCost
from utils.model_utils import to_np


class PolicyIterationLagrange(ABC):

    def __init__(self,
                 env: Union[GymEnv, str],
                 max_iter: int,
                 n_actions: int,
                 height: int,  # table length
                 width: int,  # table width
                 terminal_states: int,
                 stopping_threshold: float,
                 seed: int,
                 gamma: float = 0.99,
                 v0: float = 0.0,
                 budget: float = 0.,
                 penalty_initial_value: float = 1,
                 penalty_learning_rate: float = 0.01,
                 penalty_min_value: Optional[float] = None,
                 penalty_max_value: Optional[float] = None,
                 log_file=None,
                 ):
        super(PolicyIterationLagrange, self).__init__()
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.env = env
        self.log_file = log_file
        self.max_iter = max_iter
        self.n_actions = n_actions
        self.terminal_states = terminal_states
        self.v0 = v0
        self.seed = seed
        self.height = height
        self.width = width
        self.penalty_initial_value = penalty_initial_value
        self.penalty_min_value = penalty_min_value
        self.penalty_max_value = penalty_max_value
        self.penalty_learning_rate = penalty_learning_rate
        self.budget = budget
        self.num_timesteps = 0
        self.admissible_actions = None
        self._setup_model()

    def _setup_model(self) -> None:
        self.dual = DualVariable(self.budget,
                                 self.penalty_learning_rate,
                                 self.penalty_initial_value,
                                 min_clamp=self.penalty_min_value,
                                 max_clamp=self.penalty_max_value)
        self.v_m = self.get_init_v()
        self.pi = self.get_equiprobable_policy()

    def get_init_v(self):
        v_m = self.v0 * np.ones((self.height, self.width))
        # # Value function of terminal state must be 0
        # v0[self.e_x, self.e_y] = 0
        return v_m

    def get_equiprobable_policy(self):
        pi = 1 / self.n_actions * np.ones((self.height, self.width, self.n_actions))
        return pi

    def learn(self,
              total_timesteps: int,
              density_loss: float,
              cost_info_str: Union[str, Callable],
              latent_info_str: Union[str, Callable] = '',
              callback=None, ):
        policy_stable, dual_stable = False, False
        iter = 0
        for iter in tqdm(range(total_timesteps)):
            if policy_stable and dual_stable:
                print("\nStable at Iteration {0}.".format(iter), file=self.log_file)
                break
            self.num_timesteps += 1
            # Run the policy evaluation
            self.policy_evaluation(cost_info_str)
            # Run the policy improvement algorithm
            policy_stable = self.policy_improvement(cost_info_str)
            cumu_reward, length, dual_stable = self.dual_update(cost_info_str)
        logger.record("train/iterations", iter)
        logger.record("train/cumulative rewards", cumu_reward)
        logger.record("train/length", length)

    def step(self, action):
        return self.env.step(np.asarray([action]))

    def dual_update(self, cost_function):
        """policy rollout for recording training performance"""
        states = self.env.reset()
        cumu_reward, length = 0, 0
        actions_game, states_game, costs_game = [], [], []
        while True:
            actions, _ = self.predict(obs=states, state=None)
            actions_game.append(actions[0])
            s_primes, rewards, dones, infos = self.step(actions)
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in infos])
                if isinstance(self.env, VecNormalizeWithCost):
                    orig_costs = self.env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(states, actions)
                orig_costs = costs
            self.admissible_actions = infos[0]['admissible_actions']
            costs_game.append(orig_costs)
            states = s_primes
            states_game.append(states[0])
            done = dones[0]
            if done:
                break
            cumu_reward += rewards[0]
            length += 1
        costs_game_mean = np.asarray(costs_game).mean()
        self.dual.update_parameter(torch.tensor(costs_game_mean))
        penalty = self.dual.nu().item()
        print("Performance: dual {0}, cost: {1}, states: {2}, "
              "actions: {3}, rewards: {4}.".format(penalty,
                                                   costs_game_mean.tolist(),
                                                   np.asarray(states_game).tolist(),
                                                   np.asarray(actions_game).tolist(),
                                                   cumu_reward),
              file=self.log_file,
              flush=True)
        dual_stable = True if costs_game_mean == 0 else False
        return cumu_reward, length, dual_stable

    def policy_evaluation(self, cost_function):
        iter = 0

        delta = self.stopping_threshold + 1
        while delta >= self.stopping_threshold:
            old_v = self.v_m.copy()
            delta = 0
            # Traverse all states
            for x in range(self.height):
                for y in range(self.height):
                    # Run one iteration of the Bellman update rule for the value function
                    self.bellman_update(old_v, x, y, cost_function)
                    # Compute difference
                    delta = max(delta, abs(old_v[x, y] - self.v_m[x, y]))
            iter += 1
        print("\nThe Policy Evaluation algorithm converged after {} iterations".format(iter),
              file=self.log_file)

    def policy_improvement(self, cost_function):
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
                    s_primes, rewards, dones, infos = self.step(action)
                    # Get cost from environment.
                    if type(cost_function) is str:
                        costs = np.array([info.get(cost_function, 0) for info in infos])
                        if isinstance(self.env, VecNormalizeWithCost):
                            orig_costs = self.env.get_original_cost()
                        else:
                            orig_costs = costs
                    else:
                        costs = cost_function(states, [action])
                        orig_costs = costs
                    current_penalty = self.dual.nu().item()
                    lag_costs = current_penalty * orig_costs[0]
                    # Get value
                    curr_val = rewards[0] - lag_costs + self.gamma * self.v_m[s_primes[0][0], s_primes[0][1]]
                    # curr_val = self.v_m[s_primes[0][0], s_primes[0][1]]
                    action_values.append(curr_val)
                best_actions = np.argwhere(action_values == np.amax(action_values)).flatten().tolist()
                # Define new policy
                self.define_new_policy(x, y, best_actions)

                # Check whether the policy has changed
                if not (old_pi == self.pi[x, y, :]).all():
                    policy_stable = False

        return policy_stable

    def define_new_policy(self, x, y, best_actions):
        """Defines a new policy given the new best actions.
        Args:
            pi (array): numpy array representing the policy
            x (int): x value position of the current state
            y (int): y value position of the current state
            best_actions (list): list with best actions
            actions (list): list of every possible action
        """

        prob = 1 / len(best_actions)

        for a in range(self.n_actions):
            self.pi[x, y, a] = prob if a in best_actions else 0

    def bellman_update(self, old_v, x, y, cost_function):
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
            else:
                costs = cost_function(states, [action])
                orig_costs = costs
            gamma_values = self.gamma * old_v[s_primes[0][0], s_primes[0][1]]
            print(gamma_values)
            current_penalty = self.dual.nu().item()
            lag_costs = current_penalty * orig_costs[0]
            total += self.pi[x, y, action] * (rewards[0] - lag_costs + gamma_values)

        self.v_m[x, y] = total

    def predict(self, obs, state, deterministic=True):
        policy_prob = copy.copy(self.pi[int(obs[0][0]), int(obs[0][1])])
        if self.admissible_actions is not None:
            for c_a in range(self.n_actions):
                if c_a not in self.admissible_actions:
                    policy_prob[c_a] = -float('inf')
        best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
        action = random.choice(best_actions)
        return np.asarray([action]), state

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
        )
        torch.save(state_dict, save_path)


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

#
# def policy_iteration(self, n, p_barrier, r_barrier, v0_val, gamma, theta, seed_nr):
#     """Runs a simulation of the Policy Iteration (prediction + control) algorithm
#     Args:
#         n (int): length and width of the grid
#         p_barrier (float): probability of a cell being a barrier
#         r_barrier (int): reward for the barrier cells
#         v0_val (int): initial value for the value function
#         gamma (float): gamma parameter (between 0 and 1)
#         theta (float): threshold parameter that defines when the change in the value function is negligible
#         seed_nr (int): seed number (for reproducible results)
#     """
#
#     # Create initial environment
#     env = build_grid(n, p_barrier, r_barrier, seed_nr)
#     i = 0
#     plot_grid(env)
#
#     # Generate initial value function and policy
#     v = get_init_v(n, v0_val, env.e_x, env.e_y)
#     pi = get_equiprobable_policy(n)
#
#     # Plot initial value function and policy
#     plot_v_values(v, env.n)
#     plot_optimal_actions(env, pi)
#
#     policy_stable = False
#     while not policy_stable:
#         i += 1
#         print("\nIteration {} of Policy Iteration algorithm".format(i))
#         # Run the policy evaluation
#         policy_evaluation(env, v, pi, gamma, theta)
#         # Run the policy improvement algorithm
#         policy_stable = policy_improvement(env, v, pi, gamma)
#         plot_optimal_actions(env, pi)
#
#     print("\nPolicy Iteration algorithm converged after {} steps".format(i))
#
#
# def policy_evaluation(env, v, pi, gamma, theta):
#     """Applies the policy evaluation algorithm.
#     Args:
#         env (Environment): gridworld environment
#         v (array): numpy array representing the value function
#         pi (array): numpy array representing the policy
#         gamma (float): gamma parameter (between 0 and 1)
#         theta (float): threshold parameter that defines when the change in the value function is negligible
#     """
#
#     delta = theta + 1
#     iter = 0
#
#     while delta >= theta:
#         old_v = v.copy()
#         delta = 0
#
#         # Traverse all states
#         for x in range(env.n):
#             for y in range(env.n):
#                 # Run one iteration of the Bellman update rule for the value function
#                 bellman_update(env, v, old_v, x, y, pi, gamma)
#                 # Compute difference
#                 delta = max(delta, abs(old_v[x, y] - v[x, y]))
#
#         iter += 1
#
#     # Plot new value function
#     plot_v_values(v, env.n)
#     print("\nThe Policy Evaluation algorithm converged after {} iterations".format(iter))
#
#
# def policy_improvement(env, v, pi, gamma):
#     """Applies the Policy Improvement step.
#     Args:
#         env (Environment): gridworld environment
#         v (array): numpy array representing the value function
#         pi (array): numpy array representing the policy
#         gamma (float): gamma parameter (between 0 and 1)
#     """
#
#     policy_stable = True
#
#     # Iterate states
#     for x in range(env.n):
#         for y in range(env.n):
#             old_pi = pi[x, y, :].copy()
#
#             # Iterate all actions
#             best_actions = []
#             max_v = None
#             for a in env.actions:
#                 # Compute next state
#                 s_prime_x, s_prime_y = get_next_state(x, y, a, env.n)
#                 # Get value
#                 curr_val = env.rewards[s_prime_x, s_prime_y] + gamma * v[s_prime_x, s_prime_y]
#
#                 if max_v is None:
#                     max_v = curr_val
#                     best_actions.append(a)
#                 elif curr_val > max_v:
#                     max_v = curr_val
#                     best_actions = [a]
#                 elif curr_val == max_v:
#                     best_actions.append(a)
#
#             # Define new policy
#             define_new_policy(pi, x, y, best_actions, env.actions)
#
#             # Check whether the policy has changed
#             if not (old_pi == pi[x, y, :]).all():
#                 policy_stable = False
#
#     return policy_stable
#
#
# def bellman_update(env, v, old_v, x, y, pi, gamma):
#     """Applies the Bellman update rule to the value function
#     Args:
#         env (Environment): grid world environment
#         v (array): numpy array representing the value function
#         old_v (array): numpy array representing the value function on the last iteration
#         x (int): x value position of the current state
#         y (int): y value position of the current state
#         pi (array): numpy array representing the policy
#         gamma (float): gamma parameter (between 0 and 1)
#     """
#
#     # The value function on the terminal state always has value 0
#     if x == env.e_x and y == env.e_y:
#         return None
#
#     total = 0
#
#     for a in env.actions:
#         # Get next state
#         s_prime_x, s_prime_y = get_next_state(x, y, a, env.n)
#
#         total += pi[x, y, a] * (env.rewards[s_prime_x, s_prime_y] + gamma * old_v[s_prime_x, s_prime_y])
#
#     # Update the value function
#     v[x, y] = total
#
#
# def define_new_policy(pi, x, y, best_actions, actions):
#     """Defines a new policy given the new best actions.
#     Args:
#         pi (array): numpy array representing the policy
#         x (int): x value position of the current state
#         y (int): y value position of the current state
#         best_actions (list): list with best actions
#         actions (list): list of every possible action
#     """
#
#     prob = 1 / len(best_actions)
#
#     for a in actions:
#         pi[x, y, a] = prob if a in best_actions else 0
#
#
# def build_grid(n, p_barrier, r_barrier, seed_nr):
#     """Build an NxN grid with start and end cells, as well as some barrier cells.
#     Args:
#         n (int): length and width of the grid
#         p_barrier (float): probability of a cell being a barrier
#         r_barrier (int): reward for the barrier cells
#     Returns:
#         env (Environment): grid world environment
#     """
#
#     # Define set of possible actions: go left (0), up (1), right (2) or down (4)
#     actions = [0, 1, 2, 3]
#
#     # Define start and end cells -> these will have value 0
#     random.seed(seed_nr)
#     e_x = random.randrange(n)
#     e_y = random.randrange(n)
#
#     # Define barrier cells -> these will have barrier reward. All other have -1 reward
#     rewards = (-1) * np.ones((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i != e_x or j != e_y:
#                 p = random.uniform(0, 1)
#                 if p < p_barrier:
#                     rewards[i, j] = r_barrier
#
#     # Create environment
#     env = Environment(n, actions, rewards, e_x, e_y)
#
#     return env
#
#
# def get_next_state(x, y, a, n):
#     """Computes next state from current state and action.
#     Args:
#         x (int): x value of the current state
#         y (int): y value of the current state
#         a (int): action
#         n (int): length and width of the grid
#     Returns:
#         s_prime_x (int): x value of the next state
#         s_prime_y (int): y value of the next state
#     """
#
#     # Compute next state according to the action
#     if a == 0:
#         s_prime_x = x
#         s_prime_y = max(0, y - 1)
#     elif a == 1:
#         s_prime_x = max(0, x - 1)
#         s_prime_y = y
#     elif a == 2:
#         s_prime_x = x
#         s_prime_y = min(n - 1, y + 1)
#     else:
#         s_prime_x = min(n - 1, x + 1)
#         s_prime_y = y
#
#     return s_prime_x, s_prime_y
#
#
# def get_init_v(n, v0, e_x, e_y):
#     """Defines initial value function v_0
#     Args:
#         n (int): length and width of the grid
#         v0 (float): initial value for the value function (equal for every state)
#         e_x (int): x value of the end cell
#         e_y (int): y value of the end cell
#     Returns:
#         v0 (array): initial value function
#     """
#
#     v0 = v0 * np.ones((n, n))
#
#     # Value function of terminal state must be 0
#     v0[e_x, e_y] = 0
#
#     return v0
#
#
# def get_equiprobable_policy(n):
#     """Defines the equiprobable policy. Policy is a matrix s.t.
#         pi[x, y, a] = Pr[A = a | S = (x,y)]
#     Actions are:
#         * 0: go left
#         * 1: go up
#         * 2: go right
#         * 3: go down
#     Args:
#         n (int): length and width of the grid
#     Returns:
#         pi (array): numpy array representing the equiprobably policy
#     """
#
#     pi = 1 / 4 * np.ones((n, n, 4))
#     return pi
#
#
# def plot_grid(env):
#     """Plot grid
#     Args:
#         env (Environment): grid world environment
#     """
#
#     data = env.rewards.copy()
#     data[env.e_x, env.e_y] = 10
#
#     # create discrete colormap
#     cmap = colors.ListedColormap(['grey', 'white', 'red'])
#     bounds = [-11, -2, 0, 12]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
#
#     fig, ax = plt.subplots()
#     ax.imshow(data, cmap=cmap, norm=norm)
#
#     # draw gridlines
#     ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#     ax.set_xticks(np.arange(-.5, env.n, 1))
#     ax.set_yticks(np.arange(-.5, env.n, 1))
#
#     plt.show()
#
#
# def plot_v_values(v, n):
#     """Plots the value function in each state as a grid.
#     Args:
#         v (array): numpy array representing the value function
#         n (int):
#     """
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(v, cmap='YlOrBr', interpolation='nearest')
#
#     # draw gridlines
#     ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#     ax.set_xticks(np.arange(-.5, n, 1))
#     ax.set_yticks(np.arange(-.5, n, 1))
#
#     # Loop over data dimensions and create text annotations.
#     for i in range(n):
#         for j in range(n):
#             text = ax.text(j, i, "{:.2f}".format(v[i, j]), ha="center", va="center", color="black")
#
#     ax.set_title("Value function")
#     fig.tight_layout()
#     plt.show()
#
#
# def plot_optimal_actions(env, pi):
#     """Plots the optimal action to take in each state
#     Args:
#         env (Environment): grid world environment
#         pi (array): numpy array indicating the probability of taking each action in each state
#     """
#
#     data = env.rewards.copy()
#     data[env.e_x, env.e_y] = 10
#
#     # create discrete colormap
#     cmap = colors.ListedColormap(['grey', 'white', 'red'])
#     bounds = [-11, -2, 0, 12]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
#
#     fig, ax = plt.subplots()
#     ax.imshow(data, cmap=cmap, norm=norm)
#
#     # draw gridlines
#     ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#     ax.set_xticks(np.arange(-.5, env.n, 1))
#     ax.set_yticks(np.arange(-.5, env.n, 1))
#
#     # Loop over data dimensions and create text annotations.
#     for i in range(env.n):
#         for j in range(env.n):
#             arrow = get_arrow(pi[i, j, :])
#             text = ax.text(j, i, arrow, fontsize=16, ha="center", va="center", color="black")
#
#     ax.set_title("Policy")
#     fig.tight_layout()
#     plt.show()
#
#
# def get_arrow(prob_arr):
#     """Returns the arrows that represent the highest probability actions.
#     Args:
#         prob_arr (array): numpy array denoting the probability of taking each action on a given state
#     Returns:
#         arrow (str): string denoting the most probable action(s)
#     """
#
#     best_actions = np.where(prob_arr == np.amax(prob_arr))[0]
#     if len(best_actions) == 1:
#         if 0 in best_actions:
#             return r"$\leftarrow$"
#         if 1 in best_actions:
#             return r"$\uparrow$"
#         if 2 in best_actions:
#             return r"$\rightarrow$"
#         else:
#             return r"$\downarrow$"
#
#     elif len(best_actions) == 2:
#         if 0 in best_actions and 1 in best_actions:
#             return r"$\leftarrow \uparrow$"
#         elif 0 in best_actions and 2 in best_actions:
#             return r"$\leftrightarrow$"
#         elif 0 in best_actions and 3 in best_actions:
#             return r"$\leftarrow \downarrow$"
#         elif 1 in best_actions and 2 in best_actions:
#             return r"$\uparrow \rightarrow$"
#         elif 1 in best_actions and 3 in best_actions:
#             return r"$\updownarrow$"
#         elif 2 in best_actions and 3 in best_actions:
#             return r"$\downarrow \rightarrow$"
#
#     elif len(best_actions) == 3:
#         if 0 not in best_actions:
#             return r"$\updownarrow \rightarrow$"
#         elif 1 not in best_actions:
#             return r"$\leftrightarrow \downarrow$"
#         elif 2 not in best_actions:
#             return r"$\leftarrow \updownarrow$"
#         else:
#             return r"$\leftrightarrow \uparrow$"
#
#     else:
#         return r"$\leftrightarrow \updownarrow$"
#
#
# if __name__ == "__main__":
#     # Define arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--n", type=int, help='Width and height of the grid.')
#     parser.add_argument(
#         "--p_barrier", type=float, help='Probability of a cell being a barrier.')
#     parser.add_argument(
#         "--r_barrier", type=float, help='Reward for the barrier cells.')
#     parser.add_argument(
#         "--v0_val", type=int, help='Initial value for the value function.')
#     parser.add_argument(
#         "--gamma", type=float, help='Initial value for the value function.')
#     parser.add_argument(
#         "--theta", type=float,
#         help='Threshold parameter that defines when the change in the value function is negligible.')
#     parser.add_argument(
#         "--seed_nr", type=int, help='Seed number, for reproducible results.')
#     args = parser.parse_args()
#
#     policy_iteration(args.n, args.p_barrier, args.r_barrier, args.v0_val, args.gamma, args.theta, args.seed_nr)
