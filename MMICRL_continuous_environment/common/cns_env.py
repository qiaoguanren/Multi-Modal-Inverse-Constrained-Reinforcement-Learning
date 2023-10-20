import os
import pickle
from copy import copy, deepcopy
from typing import Any, Dict, Tuple
import numpy as np
import gym
import yaml
import cirl_stable_baselines3.common.vec_env as vec_env
from common.cns_monitor import CNSMonitor
from common.true_constraint_functions import get_true_constraint_function
from cirl_stable_baselines3.common import callbacks
from cirl_stable_baselines3.common.utils import set_random_seed
from cirl_stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv, VecNormalize, VecCostWrapper, \
    VecCostCodeWrapper
from utils.env_utils import is_mujoco, is_commonroad


def make_env(env_id, env_configs, rank, log_dir, group, multi_env=False, seed=0):
    def _init():
        # import env
        if is_commonroad(env_id):
            # import commonroad_environment.commonroad_rl.gym_commonroad
            from commonroad_environment.commonroad_rl import gym_commonroad
        elif is_mujoco(env_id):
            # from mujuco_environment.custom_envs.envs import half_cheetah
            import mujuco_environment.custom_envs
        env_configs_copy = copy(env_configs)
        if multi_env and 'commonroad' in env_id:  # update data path for the multi-env setting.
            env_configs_copy.update(
                {'train_reset_config_path': env_configs['train_reset_config_path'] + '/{0}'.format(rank)}),
        if 'external_reward' in env_configs:  # TODO: maybe rename it as constraint instead of rewards.
            del env_configs_copy['external_reward']  # this info is for the external wrapper.
        if 'constraint_id' in env_configs:
            del env_configs_copy['constraint_id']  # this info is for the external wrapper.
        if 'latent_dim' in env_configs:
            del env_configs_copy['latent_dim']  # this info is for the external wrapper.
        if 'games_by_cids' in env_configs:
            del env_configs_copy['games_by_cids']  # this info is for the external wrapper.
        env = gym.make(id=env_id,
                       **env_configs_copy)
        env.seed(seed + rank)
        del env_configs_copy
        if is_commonroad(env_id) and 'external_reward' in env_configs:
            print("Using external signal for env: {0}.".format(env_id), flush=True)
            env = CommonRoadExternalSignalsWrapper(env=env,
                                                   group=group,
                                                   **env_configs)  # wrapper_config=env_configs['external_reward']
        elif is_mujoco(env_id):
            print("Using external signal for env: {0}.".format(env_id), flush=True)
            env = MujocoExternalSignalWrapper(env=env,
                                              group=group,
                                              **env_configs)
        monitor_rank = None
        if multi_env:
            monitor_rank = rank
        env = CNSMonitor(env=env, filename=log_dir, rank=monitor_rank)
        return env

    set_random_seed(seed)
    return _init


def make_train_env(env_id, config_path, save_dir, group='PPO', base_seed=0, num_threads=1,
                   use_cost=False, normalize_obs=True, normalize_reward=True, normalize_cost=True, multi_env=False,
                   log_file=None, part_data=False,
                   **kwargs):
    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_configs = yaml.safe_load(config_file)
            if is_commonroad(env_id) and multi_env:
                env_configs['train_reset_config_path'] += '_split'
            if is_commonroad(env_id) and part_data:
                env_configs['train_reset_config_path'] += '_debug'
                env_configs['test_reset_config_path'] += '_debug'
                env_configs['meta_scenario_path'] += '_debug'
    else:
        env_configs = {}
    if 'constraint_id' in kwargs:  # the environments contain a mixture of constraints
        env_configs['constraint_id'] = kwargs['constraint_id']
        env_configs['latent_dim'] = kwargs['latent_dim']
    if 'games_by_cids' in kwargs:
        env_configs['games_by_cids'] = kwargs['games_by_cids']
    # else:
    #     env_configs['constraint_id'] = 0
    env = [make_env(env_id=env_id,
                    env_configs=env_configs,
                    rank=i,
                    log_dir=save_dir,
                    group=group,
                    multi_env=multi_env,
                    seed=base_seed)
           for i in range(num_threads)]

    env = vec_env.SubprocVecEnv(env)

    if use_cost:
        if group == 'PPO-Lag' or group == 'PI-Lag':
            env = InternalVecCostWrapper(venv=env, cost_info_str=kwargs['cost_info_str'])  # internal cost
        elif group == 'MEICRL' or group == 'InfoICRL' or group == 'ICRL' or group == 'Binary':
            env = vec_env.VecCostCodeWrapper(venv=env,
                                             latent_dim=kwargs['latent_dim'],
                                             cost_info_str=kwargs['cost_info_str'],
                                             latent_info_str=kwargs['latent_info_str'],
                                             max_seq_len=kwargs['max_seq_len'])  # cost with code
        else:
            env = vec_env.VecCostWrapper(venv=env, cost_info_str=kwargs['cost_info_str'])  # external cost

    if group == 'PPO' or group == 'GAIL':
        assert (all(key in kwargs for key in ['reward_gamma']))
        env = vec_env.VecNormalize(
            env,
            training=True,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            gamma=kwargs['reward_gamma'])
    else:
        assert (all(key in kwargs for key in ['cost_info_str', 'reward_gamma', 'cost_gamma']))
        env = vec_env.VecNormalizeWithCost(
            env, training=True,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            norm_cost=normalize_cost,
            cost_info_str=kwargs['cost_info_str'],
            reward_gamma=kwargs['reward_gamma'],
            cost_gamma=kwargs['cost_gamma'])
    return env, env_configs


def make_eval_env(env_id, config_path, save_dir, group='PPO', num_threads=1,
                  mode='test', use_cost=False, normalize_obs=True,
                  part_data=False, multi_env=False, log_file=None, **kwargs):
    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_configs = yaml.safe_load(config_file)
            if is_commonroad(env_id) and multi_env:
                env_configs['train_reset_config_path'] += '_split'
            if is_commonroad(env_id) and part_data:
                env_configs['train_reset_config_path'] += '_debug'
                env_configs['test_reset_config_path'] += '_debug'
                env_configs['meta_scenario_path'] += '_debug'
        if is_commonroad(env_id) and mode == 'test':
            env_configs["test_env"] = True
    else:
        env_configs = {}
    if 'constraint_id' in kwargs:  # the environments contain a mixture of constraints
        env_configs['constraint_id'] = kwargs['constraint_id']
        env_configs['latent_dim'] = kwargs['latent_dim']
    if 'games_by_cids' in kwargs:
        env_configs['games_by_cids'] = kwargs['games_by_cids']
    env = [make_env(env_id=env_id,
                    env_configs=env_configs,
                    rank=i,
                    group=group,
                    log_dir=os.path.join(save_dir, mode),
                    multi_env=multi_env)
           for i in range(num_threads)]
    # if mode == 'test':
    env = vec_env.DummyVecEnv(env)
    # else:
    #     env = vec_env.SubprocVecEnv(env)

    if use_cost:
        if group == 'PPO-Lag' or group == 'PI-Lag':
            env = InternalVecCostWrapper(venv=env, cost_info_str=kwargs['cost_info_str'])  # internal cost
        elif group == 'MEICRL' or group == 'InfoICRL' or group == 'ICRL' or group == 'Binary':
            env = vec_env.VecCostCodeWrapper(venv=env,
                                             latent_dim=kwargs['latent_dim'],
                                             cost_info_str=kwargs['cost_info_str'],
                                             latent_info_str=kwargs['latent_info_str'],
                                             max_seq_len=kwargs['max_seq_len'])  # cost with code
        else:
            env = vec_env.VecCostWrapper(venv=env,
                                         cost_info_str=kwargs['cost_info_str'])  # external cost, must be learned
    # print("Wrapping eval env in a VecNormalize.", file=log_file, flush=True)
    if group == 'PPO' or group == 'GAIL':
        env = vec_env.VecNormalize(env, training=False, norm_obs=normalize_obs, norm_reward=False)
    else:
        env = vec_env.VecNormalizeWithCost(env, training=False, norm_obs=normalize_obs,
                                           norm_reward=False, norm_cost=False)

    return env, env_configs


class InternalVecCostWrapper(VecEnvWrapper):
    def __init__(self, venv, cost_info_str='cost'):
        super().__init__(venv)
        self.cost_info_str = cost_info_str

    def step_async(self, actions: np.ndarray):
        self.actions = actions
        self.venv.step_async(actions)

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["cost_function"]
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        if infos is None:
            infos = {}
        # Cost depends on previous observation and current actions
        for i in range(len(infos)):
            infos[i][self.cost_info_str] = infos[i]['lag_cost']  # the pre-defined cost without learning
        self.previous_obs = obs.copy()
        return obs, rews, news, infos

    # def set_cost_function(self, cost_function):
    #     self.cost_function = cost_function

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.previous_obs = obs
        return obs

    def reset_with_values(self, info_dicts):
        """
        Reset all environments
        """
        obs = self.venv.reset_with_values(info_dicts)
        self.previous_obs = obs
        return obs

    @staticmethod
    def load(load_path: str, venv: VecEnv):
        """
        Loads a saved VecCostWrapper object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_cost_wrapper = pickle.load(file_handler)
        vec_cost_wrapper.set_venv(venv)
        return vec_cost_wrapper

    def save(self, save_path: str) -> None:
        """
        Save current VecCostWrapper object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)


# Define here to avoid circular import
def sync_envs_normalization_ppo(env: "GymEnv", eval_env: "GymEnv") -> None:
    """
    Sync eval env and train env when using VecNormalize

    :param env:
    :param eval_env:
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        if isinstance(env_tmp, VecCostWrapper) or isinstance(env_tmp, InternalVecCostWrapper) or isinstance(env_tmp,
                                                                                                            VecCostCodeWrapper):
            env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv


class SaveEnvStatsCallback(callbacks.BaseCallback):
    def __init__(
            self,
            env,
            save_path
    ):
        super(SaveEnvStatsCallback, self).__init__()
        self.env = env
        self.save_path = save_path

    def _on_step(self):
        if isinstance(self.env, vec_env.VecNormalize):
            self.env.save(os.path.join(self.save_path, "train_env_stats.pkl"))


class MujocoExternalSignalWrapper(gym.Wrapper):
    def __init__(self, env: gym.Wrapper, group: str, **wrapper_config):
        super(MujocoExternalSignalWrapper, self).__init__(env=env)
        self.wrapper_config = wrapper_config
        self.group = group
        # self.latent_dim = wrapper_config['latent_dim']

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)
        c_id = self.wrapper_config['constraint_id']
        ture_cost_function = get_true_constraint_function(env_id=self.spec.id,
                                                          env_configs=self.wrapper_config,
                                                          c_id=c_id)
        lag_cost_ture = int(ture_cost_function(obs, action) == True)

        # add true constraint for evaluation
        info.update({'lag_cost': lag_cost_ture})
        return obs, reward, done, info

    def step_with_code(self, action: np.ndarray, agent_code: np.ndarray, games_by_aids: dict) -> Tuple[
        np.ndarray, float, bool, Dict[Any, Any]]:
        # if self.wrapper_config['constraint_id'] is None:  # dynamic constraint
        #     tmp = np.argmax(action[-self.wrapper_config['latent_dim']:])
        #     constraint_id = np.argmax(action[-self.wrapper_config['latent_dim']:])
        #     action = action[:-self.wrapper_config['latent_dim']]
        # else:
        #     constraint_id = self.wrapper_config['constraint_id']  # fix constraint
        aid = np.argmax(agent_code)

        obs, reward, done, info = self.env.step(action)
        ture_cost_function = get_true_constraint_function(env_id=self.spec.id,
                                                          env_configs=self.wrapper_config,
                                                          c_id=None,  # unknown
                                                          agent_id=aid,
                                                          games_by_aids=games_by_aids,
                                                          group=self.group)
        lag_cost_ture = int(ture_cost_function(obs, action) == True)

        # add true constraint for evaluation
        info.update({'lag_cost': lag_cost_ture})
        return obs, reward, done, info


class CommonRoadExternalSignalsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Wrapper, group: str, **wrapper_config):
        super(CommonRoadExternalSignalsWrapper, self).__init__(env=env)
        self.wrapper_config = wrapper_config
        self.group = group

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        c_id = self.wrapper_config['constraint_id']
        observation, reward, done, info = self.env.step(action)
        reward_features = self.wrapper_config['external_reward']['reward_features']
        feature_bounds = self.wrapper_config['external_reward']['feature_bounds']
        feature_dims = self.wrapper_config['external_reward']['feature_dims']
        feature_penalties = self.wrapper_config['external_reward']['feature_penalties']
        terminates = self.wrapper_config['external_reward']['terminate']
        lag_cost = 0
        for idx in range(len(reward_features)):
            reward_feature = reward_features[idx]
            if reward_feature == 'velocity':
                ego_velocity_x_y = [observation[feature_dims[idx][0]], observation[feature_dims[idx][1]]]
                info["ego_velocity"] = ego_velocity_x_y
                ego_velocity = np.sqrt(np.sum(np.square(ego_velocity_x_y)))
                if ego_velocity < float(feature_bounds[idx][0]) or ego_velocity > float(feature_bounds[idx][1]):
                    reward += float(feature_penalties[idx])
                    lag_cost = 1
                    if terminates[idx]:
                        done = True
                    info.update({'is_over_speed': 1})
                else:
                    info.update({'is_over_speed': 0})
            elif reward_feature == 'same_lead_obstacle_distance':
                lanebase_relative_position = [observation[feature_dims[idx][0]]]
                info["lanebase_relative_position"] = np.asarray(lanebase_relative_position)
                _lanebase_relative_position = np.mean(lanebase_relative_position)
                if c_id == 0:
                    if _lanebase_relative_position < float(feature_bounds[idx][0]):
                        reward += float(feature_penalties[idx])
                        lag_cost = 1
                        if terminates[idx]:
                            done = True
                        info.update({'type 0 is_too_closed': 1})
                    else:
                        info.update({'type 0 is_too_closed': 0})
                else:
                    if _lanebase_relative_position < float(feature_bounds[idx + 1][0]):
                        reward += float(feature_penalties[idx])
                        lag_cost = 1
                        if terminates[idx]:
                            done = True
                        info.update({'type 1 is_too_closed': 1})
                    else:
                        info.update({'type 1 is_too_closed': 0})
            elif reward_feature == 'obstacle_distance':
                lanebase_relative_positions = [observation[feature_dims[idx][0]], observation[feature_dims[idx][1]],
                                               observation[feature_dims[idx][2]], observation[feature_dims[idx][3]],
                                               observation[feature_dims[idx][4]], observation[feature_dims[idx][5]]]
                # [p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead, p_rel_right_lead]

                info["lanebase_relative_position"] = np.asarray(lanebase_relative_positions)
                for lanebase_relative_position in lanebase_relative_positions:
                    if lanebase_relative_position < float(feature_bounds[idx][0]) or \
                            lanebase_relative_position > float(feature_bounds[idx][1]):
                        reward += float(feature_penalties[idx])
                        lag_cost = 1
                        if terminates[idx]:
                            done = True
                        info.update({'is_too_closed': 1})
                    else:
                        info.update({'is_too_closed': 0})
            else:
                raise ValueError("Unknown reward features: {0}".format(reward_feature))
        # print(ego_velocity, lag_cost)
        # if self.group == 'PPO-Lag':
        info.update({'lag_cost': lag_cost})
        return observation, reward, done, info

    def step_with_code(self, action: np.ndarray, agent_code: np.ndarray, games_by_aids: dict) -> Tuple[
        np.ndarray, float, bool, Dict[Any, Any]]:
        aid = np.argmax(agent_code)
        games_by_cids = self.wrapper_config['games_by_cids']
        vote = [0 for i in range(len(games_by_aids.keys()))]
        for game_index in games_by_aids[aid]:
            cid = games_by_cids[game_index]
            vote[cid] += 1
        c_id = np.argmax(np.asarray(vote))
        observation, reward, done, info = self.env.step(action)
        reward_features = self.wrapper_config['external_reward']['reward_features']
        feature_bounds = self.wrapper_config['external_reward']['feature_bounds']
        feature_dims = self.wrapper_config['external_reward']['feature_dims']
        feature_penalties = self.wrapper_config['external_reward']['feature_penalties']
        terminates = self.wrapper_config['external_reward']['terminate']
        lag_cost = 0
        for idx in range(len(reward_features)):
            reward_feature = reward_features[idx]
            if reward_feature == 'velocity':
                ego_velocity_x_y = [observation[feature_dims[idx][0]], observation[feature_dims[idx][1]]]
#                assert np.sum(
#                    info["ego_velocity"] - ego_velocity_x_y) == 0  # TODO: remove this line if there is an error
                info["ego_velocity"] = ego_velocity_x_y
                ego_velocity = np.sqrt(np.sum(np.square(ego_velocity_x_y)))
                if ego_velocity < float(feature_bounds[idx][0]) or ego_velocity > float(feature_bounds[idx][1]):
                    reward += float(feature_penalties[idx])
                    lag_cost = 1
                    if terminates[idx]:
                        done = True
                    info.update({'is_over_speed': 1})
                else:
                    info.update({'is_over_speed': 0})
            elif reward_feature == 'same_lead_obstacle_distance':
                lanebase_relative_position = [observation[feature_dims[idx][0]]]
                info["lanebase_relative_position"] = np.asarray(lanebase_relative_position)
                _lanebase_relative_position = np.mean(lanebase_relative_position)
                if c_id == 0:
                    if _lanebase_relative_position < float(feature_bounds[idx][0]):
                        reward += float(feature_penalties[idx])
                        lag_cost = 1
                        if terminates[idx]:
                            done = True
                        info.update({'type 0 is_too_closed': 1})
                    else:
                        info.update({'type 0 is_too_closed': 0})
                else:
                    if _lanebase_relative_position < float(feature_bounds[idx+1][0]):
                        reward += float(feature_penalties[idx])
                        lag_cost = 1
                        if terminates[idx]:
                            done = True
                        info.update({'type 1 is_too_closed': 1})
                    else:
                        info.update({'type 1 is_too_closed': 0})
            elif reward_feature == 'obstacle_distance':
                lanebase_relative_positions = [observation[feature_dims[idx][0]], observation[feature_dims[idx][1]],
                                               observation[feature_dims[idx][2]], observation[feature_dims[idx][3]],
                                               observation[feature_dims[idx][4]], observation[feature_dims[idx][5]]]
                # [p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead, p_rel_right_lead]

                info["lanebase_relative_position"] = np.asarray(lanebase_relative_positions)
                for lanebase_relative_position in lanebase_relative_positions:
                    if lanebase_relative_position < float(feature_bounds[idx][0]) or \
                            lanebase_relative_position > float(feature_bounds[idx][1]):
                        reward += float(feature_penalties[idx])
                        lag_cost = 1
                        if terminates[idx]:
                            done = True
                        info.update({'is_too_closed': 1})
                    else:
                        info.update({'is_too_closed': 0})
            else:
                raise ValueError("Unknown reward features: {0}".format(reward_feature))
        # print(ego_velocity, lag_cost)
        # if self.group == 'PPO-Lag':
        info.update({'lag_cost': lag_cost})
        return observation, reward, done, info

