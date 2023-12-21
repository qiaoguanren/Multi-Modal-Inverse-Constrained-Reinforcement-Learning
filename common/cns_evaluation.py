import os
import warnings
from abc import ABC
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn.functional as F
import gym
import numpy
import numpy as np
from matplotlib import pyplot as plt

from cirl_stable_baselines3.common.callbacks import EventCallback, BaseCallback
from cirl_stable_baselines3.common.evaluation import evaluate_policy
from cirl_stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization, VecNormalize, \
    VecNormalizeWithCost
from utils.data_utils import process_memory, build_rnn_input
from utils.model_utils import build_code

def evaluate_meicrl_cns(
        cns_model: torch.nn.Module,
):
    expert_games = [cns_model.prepare_data(cns_model.expert_obs[i], cns_model.expert_acs[i])
                    for i in range(len(cns_model.expert_obs))]
    expert_seq_games = [build_rnn_input(max_seq_length=cns_model.max_seq_length,
                                        input_data_list=expert_games[i])
                        for i in range(len(expert_games))]
    expert_seqs = torch.cat(expert_seq_games, dim=0)  # [data_num, seq_len, input_dim]
    expert_code_games = []
    expert_latent_prob_games = []
    for i in range(len(expert_seq_games)):
        expert_seq_game = expert_seq_games[i]
        rnn_batch_hidden_states = None
        for i in range(int(cns_model.max_seq_length)):
            rnn_batch_input = expert_seq_game[:, i, :]
            rnn_batch_hidden_states = cns_model.rnn(input=rnn_batch_input, hx=rnn_batch_hidden_states)
        expert_latent_prob_game = cns_model.posterior_encoder(rnn_batch_hidden_states).detach()
        expert_latent_prob_games.append(expert_latent_prob_game)
        expert_log_sum_game = torch.log(expert_latent_prob_game + cns_model.eps).sum(dim=0)
        expert_cid_game = expert_log_sum_game.argmax().repeat(len(expert_seq_game))
        expert_code_game = F.one_hot(expert_cid_game, num_classes=cns_model.latent_dim).to(cns_model.device)
        expert_code_games.append(expert_code_game)
    expert_codes = torch.cat(expert_code_games, dim=0)
    print("still working")


def evaluate_iteration_policy(
        model: ABC,
        env: Union[gym.Env, VecEnv],
        cost_function: Union[str, Callable],
        record_info_names: list,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
):
    episode_rewards, episode_nc_rewards, episode_lengths = [], [], []
    episode_costs = []
    record_infos = {}
    for record_info_name in record_info_names:
        record_infos.update({record_info_name: []})
    for i in range(n_eval_episodes):
        states = env.reset()
        cumu_reward, cumu_nc_reward, length = 0, 0, 0
        actions_game, states_game, costs_game = [], [], []
        is_constraint = [False for i in range(env.num_envs)]
        while True:
            policy_prob = model.pi[states[0][0], states[0][1]]
            action = np.argmax(policy_prob)
            actions_game.append(action)
            s_primes, rewards, dones, _infos = env.step([action])
            if 'admissible_actions' in _infos[0].keys():
                model.admissible_actions = _infos[0]['admissible_actions']
            done = dones[0]
            if done:
                break
            if type(cost_function) is str:
                costs = np.array([info.get(cost_function, 0) for info in _infos])
                if isinstance(env, VecNormalizeWithCost):
                    orig_costs = env.get_original_cost()
                else:
                    orig_costs = costs
            else:
                costs = cost_function(states, [action])
                orig_costs = costs
            episode_costs.append(costs)
            for i in range(env.num_envs):
                for record_info_name in record_info_names:
                    record_infos[record_info_name].append(np.mean(_infos[i][record_info_name]))
                if not is_constraint[i]:
                    if orig_costs[i]:
                        is_constraint[i] = True
                    else:
                        cumu_nc_reward += rewards[i]
            states = s_primes
            states_game.append(states[0])
            cumu_reward += rewards[0]
            length += 1
        episode_rewards.append(cumu_reward)
        episode_nc_rewards.append(cumu_nc_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_nc_reward = np.mean(episode_nc_rewards)
    std_nc_reward = np.std(episode_nc_rewards)
    return mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, episode_costs


def evaluate_icrl_policy(
        model: "base_class.BaseAlgorithm",
        env: Union[gym.Env, VecEnv],
        record_info_names: list,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param record_info_names: The names of recording information
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_nc_rewards, episode_lengths = [], [], []
    costs = []
    record_infos = {}
    for record_info_name in record_info_names:
        record_infos.update({record_info_name: []})
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = np.asarray([0.0] * env.num_envs)
        episode_nc_reward = np.asarray([0.0] * env.num_envs)
        is_constraint = [False for i in range(env.num_envs)]
        episode_length = 0
        obs_game = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs_game.append(obs[0])
            obs, reward, done, _info = env.step(action)
            for i in range(env.num_envs):
                if 'cost' in _info[i].keys():
                    costs.append(_info[i]['cost'])
                else:
                    costs = None
                for record_info_name in record_info_names:
                    if record_info_name == 'ego_velocity_x':
                        record_infos[record_info_name].append(np.mean(_info[i]['ego_velocity'][0]))
                    elif record_info_name == 'ego_velocity_y':
                        record_infos[record_info_name].append(np.mean(_info[i]['ego_velocity'][1]))
                    elif record_info_name == 'same_lane_leading_obstacle_distance':
                        record_infos[record_info_name].append(np.mean(_info[i]['lanebase_relative_position'][0]))
                    else:
                        record_infos[record_info_name].append(np.mean(_info[i][record_info_name]))
                if not is_constraint[i]:
                    if _info[i]['lag_cost']:
                        is_constraint[i] = True
                    else:
                        episode_nc_reward[i] += reward[i]
                episode_reward[i] += reward[i]
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        obs_game = np.asarray(obs_game)
        print(obs_game)
        episode_rewards.append(episode_reward)
        episode_nc_rewards.append(episode_nc_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_nc_reward = np.mean(episode_nc_rewards)
    std_nc_reward = np.std(episode_nc_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs


def evaluate_meicrl_policy(
        models: list("base_class.BaseAlgorithm"),
        env: Union[gym.Env, VecEnv],
        latent_dim: int,
        record_info_names: list,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        save_path=None
):
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_nc_rewards, episode_lengths = [], [], []
    costs = []
    record_infos = {}
    for record_info_name in record_info_names:
        record_infos.update({record_info_name: {}})
        for cid in range(latent_dim):
            record_infos[record_info_name].update({cid: []})

    for i in range(n_eval_episodes):
        cid = i % latent_dim
        code = build_code(code_axis=[cid for _ in range(1)],
                          code_dim=latent_dim,
                          num_envs=1)
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()

        done, state = False, None
        episode_reward = np.asarray([0.0] * env.num_envs)
        episode_nc_reward = np.asarray([0.0] * env.num_envs)
        is_constraint = [False for i in range(env.num_envs)]
        episode_length = 0
        # mem_current = process_memory()
        # print('0', mem_current)
        while not done:
            inputs = np.concatenate([obs, code], axis=1)
            action, state = models[cid].predict(inputs, state=state, deterministic=deterministic)
            obs, reward, dones, _info = env.step_with_code(actions=action, codes=code)
            if 'admissible_actions' in _info[0].keys():
                models[cid].admissible_actions = _info[0]['admissible_actions']
            done = dones[0]
            code = []
            for i in range(env.num_envs):
                code.append(_info[i]["new_code"])
                if 'cost' in _info[i].keys():
                    costs.append(_info[i]['cost'])
                else:
                    costs = None
                for record_info_name in record_info_names:
                    record_infos[record_info_name][cid].append(_info[i][record_info_name])
                if not is_constraint[i]:
                    if _info[i]['lag_cost']:
                        is_constraint[i] = True
                    else:
                        episode_nc_reward[i] += reward[i]
                episode_reward[i] += reward[i]
            code = np.asarray(code)
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        # mem_current = process_memory()
        # print('1', mem_current)
        if render:
            plt.savefig(os.path.join(save_path, "traj_visual_code-{0}.png".format(code[0])))
        # mem_current = process_memory()
        # print('2', mem_current)
        episode_rewards.append(episode_reward)
        episode_nc_rewards.append(episode_nc_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_nc_reward = np.mean(episode_nc_rewards)
    std_nc_reward = np.std(episode_nc_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs


def evaluate_with_synthetic_data(env_id, cns_model, env_configs, model_name, iteration_msg):
    if env_id == 'WGW-v0':
        map_height = int(env_configs['map_height'])
        map_width = int(env_configs['map_width'])
        for act in range(9):
            pred_cost = np.zeros([map_height, map_width])
            for i in range(map_height):
                for j in range(map_width):
                    # for k in range(act_dim-1):  # action
                    input_data = cns_model.prepare_data(obs=numpy.asarray([[i, j]]),
                                                        acs=numpy.asarray([[act]]))
                    model_output = cns_model(input_data)
                    pred_cost[i, j] += model_output
            # pred_cost /= act_dim
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            fig = plt.figure()
            shw = plt.imshow(pred_cost, cmap=cm.Greys_r)
            bar = plt.colorbar(shw)
            # plt.show()
            plt.savefig('./plot_grid_world_constraints/constraint_{0}_action-{1}_iter_{2}.png'.format(model_name, act,
                                                                                                      iteration_msg))
    pass


class CNSEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: Whether to render or not the environment during evaluation
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            best_model_save_path: str = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            callback_for_evaluate_policy: Optional[Callable] = None
    ):
        super(CNSEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.callback_for_evaluate_policy = callback_for_evaluate_policy

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                callback=self.callback_for_evaluate_policy,
                deterministic=self.deterministic,
                return_episode_rewards=True
            )
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/best_mean_reward", max(self.best_mean_reward, mean_reward))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
