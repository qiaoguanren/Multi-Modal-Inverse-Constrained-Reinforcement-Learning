import json
import logging
import os
import pickle
import time
from typing import Union, Callable

import torch
from PIL import Image
import gym
import numpy as np
import yaml
from gym import Env

from common.cns_config import get_cns_config
from common.cns_env import make_env
from common.cns_evaluation import evaluate_with_synthetic_data, evaluate_meicrl_cns
from common.cns_visualization import constraint_visualization_1d
from models.constraint_net.info_constraint_net import InfoConstraintNet
from models.constraint_net.mixture_constraint_net import MixtureConstraintNet
from models.constraint_net.variational_constraint_net import VariationalConstraintNet
from cirl_stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from cirl_stable_baselines3 import PPOLagrangian
from models.constraint_net.constraint_net import ConstraintNet
from commonroad_environment.commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv

from utils.data_utils import load_config, read_args, save_game_record, load_expert_data

# def make_env(env_id, seed,  , info_keywords=()):
#     log_dir = 'icrl/test_log'
#
#     logging_path = 'icrl/test_log'
#
#     if log_dir is not None:
#         os.makedirs(log_dir, exist_ok=True)
#
#     def _init():
#         env = gym.make(env_id, logging_path=logging_path, **env_kwargs)
#         rank = 0
#         env.seed(seed + rank)
#         log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
#         env = Monitor(env, log_file, info_keywords=info_keywords)
#         return env
#
#     return _init
from utils.env_utils import is_commonroad, is_mujoco, get_all_env_ids, get_benchmark_ids
from utils.plot_utils import pngs2gif


class CommonRoadVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.on_reset = None
        self.start_times = np.array([])

    def set_on_reset(self, on_reset_callback: Callable[[Env, float], None]):
        self.on_reset = on_reset_callback

    def reset(self):
        self.start_times = np.array([time.time()] * self.num_envs)
        return super().reset()

    def reset_benchmark(self, benchmark_ids):
        self.start_times = np.array([time.time()] * self.num_envs)
        return super().reset_benchmark(benchmark_ids)

    def step_wait(self):
        out_of_scenarios = False
        for env_idx in range(self.num_envs):
            (obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx],) = self.envs[env_idx].step(
                self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs

                # Callback
                # elapsed_time = time.time() - self.start_times[env_idx]
                # self.on_reset(self.envs[env_idx], elapsed_time)
                # self.start_times[env_idx] = time.time()

                # If one of the environments doesn't have anymore scenarios it will throw an Exception on reset()
                try:
                    obs = self.envs[env_idx].reset()
                except IndexError:
                    out_of_scenarios = True
            self._save_obs(env_idx, obs)
            self.buf_infos[env_idx]["out_of_scenarios"] = out_of_scenarios
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()


LOGGER = logging.getLogger(__name__)


def create_environments(env_id: str, viz_path: str, test_path: str, model_path: str, group: str, num_threads: int = 1,
                        normalize=True, env_kwargs=None, testing_env=False, part_data=False) -> CommonRoadVecEnv:
    """
    Create CommonRoad vectorized environment
    """
    if is_commonroad(env_id):
        if viz_path is not None:
            env_kwargs.update({"visualization_path": viz_path})
        if testing_env:
            env_kwargs.update({"play": False})
            env_kwargs["test_env"] = True
        multi_env = True if num_threads > 1 else False
        if multi_env and is_commonroad(env_id=env_id):
            env_kwargs['train_reset_config_path'] += '_split'
        if part_data and is_commonroad(env_id=env_id):
            env_kwargs['train_reset_config_path'] += '_debug'
            env_kwargs['test_reset_config_path'] += '_debug'
            env_kwargs['meta_scenario_path'] += '_debug'

    # Create environment
    envs = [make_env(env_id=env_id,
                     env_configs=env_kwargs,
                     rank=i,
                     log_dir=test_path,
                     multi_env=True if num_threads > 1 else False,
                     group=group,
                     seed=0)
            for i in range(num_threads)]
    env = CommonRoadVecEnv(envs)

    # def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
    #     # reset callback called before resetting the env
    #     if env.observation_dict["is_goal_reached"][-1]:
    #         LOGGER.info("Goal reached")
    #     else:
    #         LOGGER.info("Goal not reached")
    #     # env.render()
    #
    # env.set_on_reset(on_reset_callback)
    if normalize:
        LOGGER.info("Loading saved running average")
        vec_normalize_path = os.path.join(model_path, "train_env_stats.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            print("Loading vecnormalize.pkl from {0}".format(model_path))
        else:
            raise FileNotFoundError("vecnormalize.pkl not found in {0}".format(model_path))
        # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env


def load_model(model_path: str, iter_msg: str, log_file, device: str, group: str, **config):
    if iter_msg == 'best':
        ppo_model_path = os.path.join(model_path, "best_nominal_model")
        cns_model_path = os.path.join(model_path, "best_constraint_net_model")
    else:
        ppo_model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
        cns_model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'constraint_net')
    print('Loading ppo model from {0}'.format(ppo_model_path), flush=True, file=log_file)
    print('Loading cns model from {0}'.format(cns_model_path), flush=True, file=log_file)
    ppo_model = PPOLagrangian.load(ppo_model_path, device=device, **config)
    if group == 'ICRL' or group == 'Binary':
        cns_model = ConstraintNet.load(cns_model_path, device=device, **config)
    elif group == 'VICRL':
        cns_model = VariationalConstraintNet.load(cns_model_path, device=device, **config)
    elif group == 'MEICRL':
        cns_model = MixtureConstraintNet.load(cns_model_path, device=device, **config)
    elif group == 'PPO' or group == 'PPO-Lag':
        cns_model = None
    else:
        raise ValueError("Unknown group: {0}".format(group))
    return ppo_model, cns_model


def evaluate():
    # config, debug_mode, log_file_path = load_config(args)

    # if log_file_path is not None:
    #     log_file = open(log_file_path, 'w')
    # else:
    debug_mode = True
    log_file = None
    num_threads = 1
    if_testing_env = False

    load_model_name = 'train_MEICRL_HCWithPos-v0-multi_env-Aug-01-2022-19:24-seed_123/'
    task_name = 'MEICRL_HCWithPos-v0'
    iteration_msg = 250

    model_loading_path = os.path.join('../save_model', task_name, load_model_name)
    with open(os.path.join(model_loading_path, 'model_hyperparameters.yaml')) as reader:
        config = yaml.safe_load(reader)
    config["device"] = 'cpu'
    print(json.dumps(config, indent=4), file=log_file, flush=True)

    evaluation_path = os.path.join('../evaluate_model', config['task'], load_model_name)
    if not os.path.exists(os.path.join('../evaluate_model', config['task'])):
        os.mkdir(os.path.join('../evaluate_model', config['task']))
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    viz_path = evaluation_path
    if not os.path.exists(viz_path):
        os.mkdir(viz_path)

    if iteration_msg == 'best':
        env_stats_loading_path = model_loading_path
    else:
        env_stats_loading_path = os.path.join(model_loading_path, 'model_{0}_itrs'.format(iteration_msg))
    if config['env']['config_path'] is not None:
        with open(config['env']['config_path'], "r") as config_file:
            env_configs = yaml.safe_load(config_file)
    else:
        env_configs = {}
    env = create_environments(env_id=config['env']['train_env_id'],
                              viz_path=viz_path,
                              test_path=evaluation_path,
                              model_path=env_stats_loading_path,
                              group=config['group'],
                              num_threads=num_threads,
                              normalize=not config['env']['dont_normalize_obs'],
                              env_kwargs=env_configs,
                              testing_env=if_testing_env,
                              part_data=debug_mode)
    # Load expert data
    (expert_obs, expert_acs, expert_rs, expert_cs), expert_mean_reward = load_expert_data(
        expert_path=config['running']['expert_path'].replace('expert_data/', 'expert_data/debug_') if debug_mode else
        config['running']['expert_path'],
        num_rollouts=config['running']['expert_rollouts'],
        store_by_game=config['running']['store_by_game'],
        add_latent_code=True,
        log_file=log_file
    )
    cn_parameters = get_cns_config(config=config,
                                   train_env=env,
                                   expert_obs=expert_obs,
                                   expert_acs=expert_acs,
                                   log_file=log_file)
    if 'InfoICRL' == config['group']:
        cn_parameters.update({'latent_dim': config['CN']['latent_dim'], })
        constraint_net = InfoConstraintNet(**cn_parameters)
    elif 'MEICRL' == config['group']:
        cn_parameters.update({'latent_dim': config['CN']['latent_dim'], })
        cn_parameters.update({'max_seq_length': config['running']['max_seq_length'], })
        constraint_net = MixtureConstraintNet(**cn_parameters)
    else:
        raise ValueError("Unknown group: {0}".format(config['group']))
    if str(iteration_msg) == 'best':
        cns_model_path = os.path.join(model_loading_path, "best_constraint_net_model")
    else:
        cns_model_path = os.path.join(model_loading_path, 'model_{0}_itrs'.format(str(iteration_msg)), 'constraint_net')
    print('Loading cns model from {0}'.format(cns_model_path), flush=True, file=log_file)
    state_dict = torch.load(cns_model_path)
    constraint_net.constraint_functions.load_state_dict(state_dict["cn_network"])
    evaluate_meicrl_cns(
        cns_model=constraint_net
    )

if __name__ == '__main__':
    # args = read_args()
    evaluate()
