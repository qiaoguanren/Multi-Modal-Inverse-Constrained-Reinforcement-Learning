import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Union, Callable
import numpy as np
import yaml
cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
sys.path.append('../')
from cirl_stable_baselines3 import PPOLagrangian
from cirl_stable_baselines3.iteration import PolicyIterationLagrange
from cirl_stable_baselines3.iteration.policy_interation_lag import load_pi
from cirl_stable_baselines3.ppo_lag.ma_ppo_lag import MultiAgentPPOLagrangian
from utils.model_utils import load_ppo_config, load_policy_iteration_config


from gym import Env
from common.cns_env import make_env
from cirl_stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from utils.data_utils import load_config, read_args, save_game_record, load_ppo_model
from utils.env_utils import get_all_env_ids, get_benchmark_ids, is_mujoco, is_commonroad


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


def create_environments(env_id: str, config_path: str, viz_path: str, test_path: str, model_path: str, group: str,
                        num_threads: int = 1,
                        normalize=True, testing_env=False, part_data=False,
                        constraint_id=0, latent_dim=2, max_seq_len=None, ):
    """
    Create CommonRoad vectorized environment
    """
    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_kwargs = yaml.safe_load(config_file)
    else:
        env_kwargs = {}
    if is_commonroad(env_id):
        if viz_path is not None:
            env_kwargs.update({"visualization_path": viz_path})
        if testing_env:
            env_kwargs.update({"play": False})
            env_kwargs["test_env"] = True
        multi_env = True if num_threads > 1 else False
        if multi_env:
            env_kwargs['train_reset_config_path'] += '_split'
        if part_data:
            env_kwargs['train_reset_config_path'] += '_debug'
            env_kwargs['test_reset_config_path'] += '_debug'
            env_kwargs['meta_scenario_path'] += '_debug'

    env_kwargs['constraint_id'] = constraint_id  # the environments contain a mixture of constraints
    env_kwargs['latent_dim'] = latent_dim

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

    return env, env_kwargs


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_mode", help="whether to use the debug mode",
                        dest="DEBUG_MODE",
                        default=False, required=False)
    parser.add_argument("-n", "--num_threads", help="number of threads for loading envs.",
                        dest="NUM_THREADS",
                        default=1, required=False)
    parser.add_argument("-mn", "--model_name", help="name of the model to be loaded.",
                        dest="MODEL_NAME",
                        default=None, required=True)
    parser.add_argument("-tn", "--task_name", help="name of the task for the model.",
                        dest="TASK_NAME",
                        default=None, required=True)
    parser.add_argument("-ct", "--constraint_type", help="the constraint to be followed by the generated data.",
                        dest="CONSTRAINT_TYPE",
                        default=None, required=True)
    parser.add_argument("-ld", "--latent_code_dimension", help="the dimension of latent codes.",
                        dest="LATENT_DIMENSION",
                        default=1, required=True)
    parser.add_argument("-nsy", "--noisy_demonstration", help="generate the noisy demonstration data.",
                        dest="NOISY_DEMO",
                        default=0, required=True)
    args = parser.parse_args()
    debug_mode = args.DEBUG_MODE
    num_threads = int(args.NUM_THREADS)
    load_model_name = args.MODEL_NAME
    task_name = args.TASK_NAME
    data_generate_type = args.CONSTRAINT_TYPE
    noisy_demonstration = bool(int(args.NOISY_DEMO))
    log_file = None
    # load_model_name = 'train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-06-2022-11:29-seed_123'
    # task_name = 'PPO-highD'
    # data_generate_type = 'no-over_speed'
    iteration_msg = 'best'
    if_testing_env = False
    if debug_mode:
        pass

    model_loading_path = os.path.join('../save_model', task_name, load_model_name)
    with open(os.path.join(model_loading_path, 'model_hyperparameters.yaml')) as reader:
        config = yaml.safe_load(reader)
    print(json.dumps(config, indent=4), file=log_file, flush=True)

    constraint_id = config['env']['constraint_id']
    code = np.zeros((int(args.LATENT_DIMENSION)), dtype=np.int)
    if noisy_demonstration:
        for i in range(int(args.LATENT_DIMENSION)):
            if i == constraint_id:
                continue
            else:
                code[i] = 1
    else:
        code[constraint_id] = 1

    evaluation_path = os.path.join('../evaluate_model', config['task'], load_model_name)
    if not os.path.exists(os.path.join('../evaluate_model', config['task'])):
        os.mkdir(os.path.join('../evaluate_model', config['task']))
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)

    save_expert_data_path = os.path.join('../data/expert_data/', '{3}{0}{1}_{2}'.format(
        'debug_' if debug_mode else '',
        data_generate_type,
        load_model_name,
        'noisy_' if noisy_demonstration else '',
    ))
    if not os.path.exists(save_expert_data_path):
        os.mkdir(save_expert_data_path)
    if iteration_msg == 'best':
        env_stats_loading_path = model_loading_path
    else:
        env_stats_loading_path = os.path.join(model_loading_path, 'model_{0}_itrs'.format(iteration_msg))

    env, env_configs = create_environments(env_id=config['env']['train_env_id'],
                                           config_path=config['env']['config_path'],
                                           viz_path=None,
                                           test_path=evaluation_path,
                                           model_path=env_stats_loading_path,
                                           group=config['group'],
                                           num_threads=num_threads,
                                           normalize=not config['env']['dont_normalize_obs'],
                                           testing_env=if_testing_env,
                                           part_data=debug_mode,
                                           constraint_id=config['env']['constraint_id'],
                                           latent_dim=int(args.LATENT_DIMENSION),
                                           )

    # TODO: this is for a quick check, maybe remove it in the future
    env.norm_reward = False
    if not noisy_demonstration:
        if "PPO" in config['group']:
            model = load_ppo_model(model_loading_path, iter_msg=iteration_msg, log_file=log_file)
        elif "PI" in config['group']:
            model = load_pi(model_loading_path, iter_msg=iteration_msg, log_file=log_file)
        else:
            raise ValueError("Unknown model {0}.".format(config['group']))
    else:
        if "PPO" in config['group']:
            ppo_parameters = load_ppo_config(config=config,
                                             train_env=env,
                                             seed=int(model_loading_path.split('_')[-1]),
                                             log_file=log_file)
            create_ppo_agent = lambda: PPOLagrangian(**ppo_parameters)
            model = create_ppo_agent()
        elif "PI" in config['group']:
            iteration_parameters = load_policy_iteration_config(config=config,
                                                                env_configs=env_configs,
                                                                train_env=env,
                                                                seed=int(model_loading_path.split('_')[-1]),
                                                                log_file=log_file)
            create_iteration_agent = lambda: PolicyIterationLagrange(**iteration_parameters)
            model = create_iteration_agent()
        else:
            raise ValueError("Unknown model {0}.".format(config['group']))
        print("Skip model loading for noisy demonstration.", file=log_file)
    total_scenarios, benchmark_idx = 0, 0
    if is_commonroad(env_id=config['env']['train_env_id']):
        max_benchmark_num, env_ids, benchmark_total_nums = get_all_env_ids(num_threads, env)
        # num_collisions, num_off_road, num_goal_reaching, num_timeout = 0, 0, 0, 0
    elif is_mujoco(env_id=config['env']['train_env_id']):
        max_benchmark_num = 70 / num_threads  # max number of expert traj is 50 for mujoco
    else:
        raise ValueError("Unknown env_id: {0}".format(config['env']['train_env_id']))

    success = 0
    saved_num = 0
    benchmark_id_all = []
    while benchmark_idx < max_benchmark_num:
        if is_commonroad(env_id=config['env']['train_env_id']):
            benchmark_ids = get_benchmark_ids(num_threads=num_threads, benchmark_idx=benchmark_idx,
                                              benchmark_total_nums=benchmark_total_nums, env_ids=env_ids)
            benchmark_num_per_step = len(benchmark_ids)
            obs = env.reset_benchmark(benchmark_ids=benchmark_ids)
        elif is_mujoco(env_id=config['env']['train_env_id']):
            benchmark_ids = [i for i in range((benchmark_idx) * num_threads, (benchmark_idx + 1) * num_threads)]
            benchmark_num_per_step = num_threads
            obs = env.reset()
        else:
            raise ValueError("Unknown env_id: {0}".format(config['env']['train_env_id']))
        done, state = False, None
        # benchmark_ids = [env.venv.envs[i].benchmark_id for i in range(num_threads)]
        not_duplicate = [True for i in range(num_threads)]
        if is_commonroad(env_id=config['env']['train_env_id']):
            for b_idx in range(benchmark_num_per_step):
                benchmark_id = benchmark_ids[b_idx]
                if benchmark_id in benchmark_id_all:
                    print('skip game', benchmark_id, file=log_file, flush=True)
                    not_duplicate[b_idx] = False
                else:
                    benchmark_id_all.append(benchmark_id)
                    print('senario id', benchmark_id, file=log_file, flush=True)

        obs_all = [[] for i in range(num_threads)]
        original_obs_all = [[] for i in range(num_threads)]
        action_all = [[] for i in range(num_threads)]
        reward_all = [[] for i in range(num_threads)]
        reward_sums = [0 for i in range(num_threads)]
        running_steps = [0 for i in range(num_threads)]
        multi_thread_dones = [False for i in range(num_threads)]
        infos_done = [None for i in range(num_threads)]
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            original_obs = env.get_original_obs() if isinstance(env, VecNormalize) else obs
            new_obs, rewards, dones, infos = env.step(action)

            for i in range(num_threads):
                if not multi_thread_dones[i]:
                    obs_all[i].append(obs[i])
                    original_obs_all[i].append(original_obs[i])
                    action_all[i].append(action[i])
                    reward_all[i].append(rewards[i])
                    running_steps[i] += 1
                    reward_sums[i] += rewards[i]
                    if dones[i]:
                        infos_done[i] = infos[i]
                        multi_thread_dones[i] = True
            # save_game_record(info[0], game_info_file)
            done = True
            for multi_thread_done in multi_thread_dones:
                if not multi_thread_done:
                    done = False
                    break
            obs = new_obs

        for i in range(num_threads):
            if not not_duplicate[i]:
                continue
            total_scenarios += 1
            info = infos_done[i]
            termination_reasons = []
            if is_commonroad(env_id=config['env']['train_env_id']):
                if info["episode"].get("is_time_out", 0) == 1:
                    termination_reasons.append("time_out")
                if info["episode"].get("is_off_road", 0) == 1:
                    termination_reasons.append("off_road")
                if info["episode"].get("is_collision", 0) == 1:
                    termination_reasons.append("collision")
                if info["episode"].get("is_goal_reached", 0) == 1:
                    termination_reasons.append("goal_reached")
                if "is_over_speed" in info["episode"].keys() and info["episode"].get("is_over_speed", 0) == 1:
                    termination_reasons.append("over_speed")
                if "is_too_closed" in info["episode"].keys() and info["episode"].get("is_too_closed", 0) == 1:
                    termination_reasons.append("too_closed")
            elif is_mujoco(env_id=config['env']['train_env_id']):
                if info["episode"].get('constraint', 0) == 1:
                    termination_reasons.append("constraint")
                else:
                    termination_reasons.append("Game Finished")
            save_constraint_expert = True
            for termination_reason in termination_reasons:
                if termination_reason in data_generate_type:
                    save_constraint_expert = False

            if save_constraint_expert:
                saved_num += 1
                print('saving expert data for game {0} with '
                      'terminal reason {1} and reward sum {2}.'.format(benchmark_ids[i],
                                                                       termination_reasons,
                                                                       reward_sums[i]),
                      file=log_file, flush=True)

                saving_expert_data = {
                    'observations': np.asarray(obs_all[i]),
                    'actions': np.asarray(action_all[i]),
                    'original_observations': np.asarray(original_obs_all[i]),
                    'codes': np.asarray([code] * len(obs_all[i])),
                    'reward': np.asarray(reward_all[i]),
                    'reward_sum': reward_sums[i]
                }
                print(np.asarray(obs_all[i]))
                if is_commonroad(env_id=config['env']['train_env_id']):
                    with open(os.path.join(save_expert_data_path,
                                           '{3}scene-{0}_code-{1}_len-{2}{3}.pkl'.format(benchmark_ids[i],
                                                                                         list(code),
                                                                                         running_steps[i],
                                                                                         '-noisy' if noisy_demonstration else '', ),
                                           ), 'wb') as file:
                        # A new file will be created
                        pickle.dump(saving_expert_data, file)
                elif is_mujoco(env_id=config['env']['train_env_id']):
                    with open(os.path.join(save_expert_data_path,
                                           'scene-{0}_code-{1}_len-{2}{3}.pkl'.format(success,
                                                                                      list(code),
                                                                                      running_steps[i],
                                                                                      '-noisy' if noisy_demonstration else '', )
                                           ), 'wb') as file:
                        # A new file will be created
                        pickle.dump(saving_expert_data, file)
                else:
                    raise ValueError("Unknown env :{0}".format(config['env']['train_env_id']))
            else:
                print('Deleting expert data for game {0} with terminal reason: {1}'.format(benchmark_ids[i],
                                                                                           termination_reasons),
                      file=log_file, flush=True)

            if is_commonroad(config['env']['train_env_id']) and "goal_reached" in termination_reasons:
                print('{0}: goal reached'.format(benchmark_ids[i]), file=log_file, flush=True)
                success += 1
            elif is_mujoco(config['env']['train_env_id']) and save_constraint_expert:
                success += 1
            # if not info["out_of_scenarios"]:
            #     out_of_scenarios = False
        benchmark_idx += 1

    print('total', total_scenarios, 'success', success, 'saved_num', saved_num, file=log_file, flush=True)


if __name__ == '__main__':
    # args = read_args()
    run()
