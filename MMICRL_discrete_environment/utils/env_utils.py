import os
import numpy as np
import cirl_stable_baselines3.common.vec_env as vec_env
from cirl_stable_baselines3.common.callbacks import BaseCallback


def get_benchmark_ids(num_threads, benchmark_idx, benchmark_total_nums, env_ids):
    benchmark_ids = []
    for i in range(num_threads):
        if benchmark_total_nums[i] > benchmark_idx:
            benchmark_ids.append(env_ids[i][benchmark_idx])
        else:
            benchmark_ids.append(None)
    return benchmark_ids


def multi_threads_sample_from_agent(agent, env, rollouts, num_threads, store_by_game=False):
    # if isinstance(env, vec_env.VecEnv):
    #     assert env.num_envs == 1, "You must pass only one environment when using this function"
    rollouts = int(float(rollouts) / num_threads)
    all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
    sum_rewards, all_lengths = [], []
    max_benchmark_num, env_ids, benchmark_total_nums = get_all_env_ids(num_threads, env)
    assert rollouts <= min(benchmark_total_nums)
    for j in range(rollouts):
        benchmark_ids = get_benchmark_ids(num_threads=num_threads, benchmark_idx=j,
                                          benchmark_total_nums=benchmark_total_nums, env_ids=env_ids)
        obs = env.reset_benchmark(benchmark_ids=benchmark_ids)  # force reset for all games
        multi_thread_already_dones = [False for i in range(num_threads)]
        done, states = False, None
        episode_sum_rewards = [0 for i in range(num_threads)]
        episode_lengths = [0 for i in range(num_threads)]
        origin_obs_game = [[] for i in range(num_threads)]
        obs_game = [[] for i in range(num_threads)]
        acs_game = [[] for i in range(num_threads)]
        rs_game = [[] for i in range(num_threads)]
        while not done:
            actions, states = agent.predict(obs, state=states, deterministic=False)
            original_obs = env.get_original_obs()
            new_obs, rewards, dones, _infos = env.step(actions)
            # benchmark_ids = [env.venv.envs[i].benchmark_id for i in range(num_threads)]
            # print(benchmark_ids)
            for i in range(num_threads):
                if not multi_thread_already_dones[i]:  # we will not store when a game is done
                    origin_obs_game[i].append(original_obs[i])
                    obs_game[i].append(obs[i])
                    acs_game[i].append(actions[i])
                    rs_game[i].append(rewards[i])
                    episode_sum_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                if dones[i]:
                    multi_thread_already_dones[i] = True
            done = True
            for multi_thread_done in multi_thread_already_dones:
                if not multi_thread_done:  # we will wait for all games done
                    done = False
                    break
            obs = new_obs
        origin_obs_game = [np.array(origin_obs_game[i]) for i in range(num_threads)]
        obs_game = [np.array(obs_game[i]) for i in range(num_threads)]
        acs_game = [np.array(acs_game[i]) for i in range(num_threads)]
        rs_game = [np.array(rs_game[i]) for i in range(num_threads)]
        all_orig_obs += origin_obs_game
        all_obs += obs_game
        all_acs += acs_game
        all_rs += rs_game

        sum_rewards += episode_sum_rewards
        all_lengths += episode_lengths
    if not store_by_game:
        all_orig_obs = np.concatenate(all_orig_obs, axis=0)
        all_obs = np.concatenate(all_obs, axis=0)
        all_acs = np.concatenate(all_acs, axis=0)
        all_rs = np.concatenate(all_rs, axis=0)
    return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, all_lengths

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))


def get_all_env_ids(num_threads, env):
    max_benchmark_num = 0
    env_ids = []
    benchmark_total_nums = []
    for i in range(num_threads):
        try:  # we need to change this setting if you modify the number of env wrappers.
            env_ids.append(list(env.venv.envs[i].env.env.env.all_problem_dict.keys()))
        except:
            env_ids.append(list(env.venv.envs[i].env.env.all_problem_dict.keys()))
        benchmark_total_nums.append(len(env_ids[i]))
        if len(env_ids[i]) > max_benchmark_num:
            max_benchmark_num = len(env_ids[i])
    return max_benchmark_num, env_ids, benchmark_total_nums


def is_mujoco(env_id):
    mujoco_env_id = ['HC', 'AntWall', 'Pendulum', 'Walker', 'LGW', 'WGW', 'Swimmer', 'Circle']
    # if 'HC' in env_id or 'LGW' in env_id or 'AntWall' in env_id or 'Pendulum' in env_id or 'Walker' in env_id:
    for item in mujoco_env_id:
        if item in env_id:
            return True
    return False
    # if isinstance(env, mujoco_env.MujocoEnv):
    #     return True
    # else:
    #     return False


def is_commonroad(env_id):
    if 'commonroad' in env_id:
        return True
    else:
        return False


def get_obs_feature_names(env, env_id):
    if is_commonroad(env_id):
        # try:  # we need to change this setting if you modify the number of env wrappers.
        #     observation_space_dict = env.venv.envs[0].env.env.env.observation_collector.observation_space_dict
        # except:
        #     observation_space_dict = env.venv.envs[0].env.env.observation_collector.observation_space_dict
        # observation_space_names = observation_space_dict.keys()
        # for key in observation_space_names:
        #     feature_len = observation_space_dict[key].shape[0]
        #     for i in range(feature_len):
        #         feature_names.append(key + '_' + str(i))
        feature_names = \
            ['distance_goal_long_0',
             'distance_goal_long_advance_0',
             'distance_goal_lat_0',
             'distance_goal_lat_advance_0',
             'is_goal_reached_0',
             'is_time_out_0',
             'v_ego_0', 'v_ego_1',
             'a_ego_0', 'a_ego_1',
             'is_friction_violation_0',
             'remaining_steps_0',
             'lane_based_v_rel_0', 'lane_based_v_rel_1', 'lane_based_v_rel_2', 'lane_based_v_rel_3',
             'lane_based_v_rel_4', 'lane_based_v_rel_5',
             'lane_based_p_rel_0', 'lane_based_p_rel_1', 'lane_based_p_rel_2', 'lane_based_p_rel_3',
             'lane_based_p_rel_4', 'lane_based_p_rel_5',
             'vehicle_type_0', 'vehicle_type_1', 'vehicle_type_2', 'vehicle_type_3', 'vehicle_type_4', 'vehicle_type_5',
             'is_collision_0',
             'is_off_road_0',
             'left_marker_distance_0',
             'right_marker_distance_0',
             'left_road_edge_distance_0',
             'right_road_edge_distance_0',
             'lat_offset_0',
             'lane_curvature_0',
             'route_reference_path_positions_0', 'route_reference_path_positions_1', 'route_reference_path_positions_2',
             'route_reference_path_positions_3', 'route_reference_path_positions_4', 'route_reference_path_positions_5',
             'route_reference_path_positions_6', 'route_reference_path_positions_7', 'route_reference_path_positions_8',
             'route_reference_path_positions_9',
             'route_reference_path_orientations_0', 'route_reference_path_orientations_1',
             'route_reference_path_orientations_2', 'route_reference_path_orientations_3',
             'route_reference_path_orientations_4',
             'route_multilanelet_waypoints_positions_0', 'route_multilanelet_waypoints_positions_1',
             'route_multilanelet_waypoints_positions_2', 'route_multilanelet_waypoints_positions_3',
             'route_multilanelet_waypoints_positions_4', 'route_multilanelet_waypoints_positions_5',
             'route_multilanelet_waypoints_positions_6', 'route_multilanelet_waypoints_positions_7',
             'route_multilanelet_waypoints_positions_8', 'route_multilanelet_waypoints_positions_9',
             'route_multilanelet_waypoints_positions_10', 'route_multilanelet_waypoints_positions_11',
             'route_multilanelet_waypoints_orientations_0', 'route_multilanelet_waypoints_orientations_1',
             'route_multilanelet_waypoints_orientations_2', 'route_multilanelet_waypoints_orientations_3',
             'route_multilanelet_waypoints_orientations_4', 'route_multilanelet_waypoints_orientations_5',
             'distance_togoal_via_referencepath_0', 'distance_togoal_via_referencepath_1',
             'distance_togoal_via_referencepath_2']
        return feature_names
    elif is_mujoco(env_id):
        if env_id == 'Circle-v0':
            feature_names = ["x_t-4", "y_t-4", "x_t-3", "y_t-3", "x_t-2", "y_t-2", "x_t-1", "y_t-1", "x_t", "y_t", ]
        else:
            feature_names = ['(pls visit mujoco xml settings at: {0})'.format(
                'https://www.gymlibrary.ml/environments/mujoco/')]
        return feature_names


def check_if_duplicate_seed(seed, config, current_time_date, save_model_mother_dir, log_file, max_endure_date=1):
    from datetime import datetime as dt
    if is_mujoco(env_id=config['env']['train_env_id']):
        task_saving_path = '{0}/{1}/'.format(config['env']['save_dir'], config['task'])
        if not os.path.exists(task_saving_path):
            skip_running = False
        else:
            all_candidate_seeds = [123, 321, 456, 654, 666]
            current_save_date = dt.strptime(current_time_date, "%b-%d-%Y-%H:%M")
            exist_seeds = set()
            assert seed in all_candidate_seeds
            for save_file_name in os.listdir(task_saving_path):
                task_mother_dir = task_saving_path + save_file_name
                # tmp = save_model_mother_dir.split('-seed_')[0].replace(current_time_date, '')
                if save_model_mother_dir.split('-seed_')[0].replace(current_time_date, '') in task_mother_dir:
                    file_seed = int(save_file_name.split('-seed_')[1])
                    file_date = \
                        task_mother_dir.replace(save_model_mother_dir.split('-seed_')[0].replace(current_time_date, ''),
                                                '').split('-seed_')[0]
                    assert file_seed in all_candidate_seeds
                    try:
                        pass_save_date = dt.strptime(file_date, "%b-%d-%Y-%H:%M")
                        diff_date = current_save_date - pass_save_date
                        if diff_date.days <= max_endure_date:
                            exist_seeds.add(file_seed)
                    except Exception as e:
                        print(e)
            exist_seeds = sorted(list(exist_seeds))
            print("existing running seeds are : {0}".format(exist_seeds), flush=True, file=log_file)
            if seed in exist_seeds and exist_seeds.index(seed) < len(exist_seeds) - 1:
                skip_running = True
                print("Skipping running for seed {0}".format(seed), flush=True, file=log_file)
            else:
                skip_running = False
    else:
        skip_running = False
    return skip_running
