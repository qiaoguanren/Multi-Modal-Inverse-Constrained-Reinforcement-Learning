import argparse
import copy
import os
import pickle
import shutil
from collections import deque
import time
import pickle5
import psutil
import torch
import yaml
import numpy as np
from gym.utils.colorize import color2num
from tqdm import tqdm
import cirl_stable_baselines3.common.callbacks as callbacks
from cirl_stable_baselines3 import PPO
from cirl_stable_baselines3.common.utils import safe_mean


def load_config(args=None):
    assert os.path.exists(args.config_file), "Invalid configs file {0}".format(args.config_file)
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    return config, args.DEBUG_MODE, args.LOG_FILE_PATH, args.PART_DATA, int(args.NUM_THREADS), int(args.SEED)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to configs file")
    # parser.add_argument("-t", "--train_flag", help="if training",
    #                     dest="TRAIN_FLAG",
    #                     default='1', required=False)
    parser.add_argument("-d", "--debug_mode", help="whether to use the debug mode",
                        dest="DEBUG_MODE",
                        default=False, required=False)
    parser.add_argument("-p", "--part_data", help="whether to use the partial dataset",
                        dest="PART_DATA",
                        default=False, required=False)
    parser.add_argument("-n", "--num_threads", help="number of threads for loading envs.",
                        dest="NUM_THREADS",
                        default=1, required=False)
    parser.add_argument("-s", "--seed", help="the seed of randomness",
                        dest="SEED",
                        default=123,
                        required=False,
                        type=int)
    parser.add_argument("-l", "--log_file", help="log file", dest="LOG_FILE_PATH", default=None, required=False)
    args = parser.parse_args()
    return args


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


# This callback should be used with the 'with' block, to allow for correct
# initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = int(total_timesteps)

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps, dynamic_ncols=True)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


# =============================================================================
# Custom callbacks
# =============================================================================

class ProgressBarCallback(callbacks.BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = int(self.num_timesteps)
        self._pbar.update(0)

    def _on_rollout_end(self):
        total_reward = safe_mean([ep_info["reward"] for ep_info in self.model.ep_info_buffer])
        try:
            average_cost = safe_mean(self.model.rollout_buffer.orig_costs)
            total_cost = np.sum(self.model.rollout_buffer.orig_costs)
            self._pbar.set_postfix(
                tr='%05.1f' % total_reward,
                ac='%05.3f' % average_cost,
                tc='%05.1f' % total_cost,
                nu='%05.1f' % self.model.dual.nu().item()
            )
        except:  # No cost
            # average_cost = 0
            # total_cost = 0
            # desc = "No Cost !!!"
            self._pbar.set_postfix(
                tr='%05.1f' % total_reward,
                ac='No Cost',
                tc='No Cost',
                nu='No Dual'
            )


def del_and_make(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)


# def compute_moving_average(result_all, average_num=100):
#     result_moving_average_all = []
#     moving_values = deque([], maxlen=average_num)
#     for result in result_all:
#         moving_values.append(result)
#         if len(moving_values) < average_num:  # this is to average the results in the beginning
#             result_moving_average_all.append(np.mean(result_all[:100]))
#         else:
#             result_moving_average_all.append(np.mean(moving_values))
#     return np.asarray(result_moving_average_all)


def compute_moving_average(result_all, average_num=100):
    if len(result_all) <= average_num:
        average_num = len(result_all)
    result_moving_all = []

    for i in range(average_num):
        # tmp = result_all[len(result_all)-i:]
        filling_in_values = np.random.choice(result_all[-i:], i)
        result_moving_all.append(np.concatenate([result_all[i:], filling_in_values]))
    result_moving_all = np.mean(result_moving_all, axis=0)
    return result_moving_all[:-average_num]


def read_running_logs_by_cid(monitor_path_all, read_keys, max_episodes, max_reward, min_reward, cid_num):
    # handle the keys
    with open(monitor_path_all[0], 'r') as file:
        running_logs = file.readlines()
    key_indices = {}
    record_keys = running_logs[1].replace('\n', '').split(',')
    try:
        aid_index = record_keys.index('a_id')
    except:  # TODO: remove it later
        aid_index = record_keys.index('c_id')
    for key in read_keys:
        key_idx = record_keys.index(key)
        key_indices.update({key: key_idx})
    read_running_logs_by_cid = {}
    for cid in range(cid_num):
        read_running_logs = {}
        for key in read_keys:
            read_running_logs.update({key: []})
        read_running_logs_by_cid.update({cid: read_running_logs})

    # read all the logs
    running_logs_all = []
    max_len = 0
    for monitor_path in monitor_path_all:
        with open(monitor_path, 'r') as file:
            running_logs = file.readlines()
        running_logs_all.append(running_logs[2:])
        if len(running_logs[2:]) > max_len:
            max_len = len(running_logs[2:])
    max_len = min(float(max_episodes / len(monitor_path_all)), max_len)

    # iteratively read the logs
    line_num = 0
    while line_num < max_len:
        # old_results = None
        for i in range(len(monitor_path_all)):
            if line_num >= len(running_logs_all[i]):
                continue
            running_performance = running_logs_all[i][line_num]
            log_items = running_performance.split(',')
            results = [item.replace("\n", "") for item in log_items]
            cid = int(results[aid_index])
            # print(cid)
            for key in read_keys:
                read_running_logs_by_cid[cid][key].append(float(results[key_indices[key]]))
        line_num += 1

    return read_running_logs_by_cid


def save_game_record(info, file, type, cost=None):
    if type == 'commonroad':
        is_collision = info["is_collision"]
        is_time_out = info["is_time_out"]
        is_off_road = info["is_off_road"]
        record_extra_info = []
        if 'ego_velocity' in info.keys():
            ego_velocity_x_y = info["ego_velocity"]
            ego_velocity_x = ego_velocity_x_y[0]
            ego_velocity_y = ego_velocity_x_y[1]
            record_extra_info.append("{0:.3f}, {1:.3f}".format(ego_velocity_x, ego_velocity_y))
        if 'lanebase_relative_position' in info.keys():
            lanebase_relative_position = info["lanebase_relative_position"]
            record_extra_info.append("{0:.3f}".format(lanebase_relative_position[0]))
        record_extra_info_str = ','.join(record_extra_info)
        is_goal_reached = info["is_goal_reached"]
        current_step = info["current_episode_time_step"]
        file.write("{0}, {1}, {2}, {3:.0f}, {4:.0f}, {5:.0f}, {6:.0f}\n".format(current_step,
                                                                                record_extra_info_str,
                                                                                cost,
                                                                                is_collision,
                                                                                is_off_road,
                                                                                is_goal_reached,
                                                                                is_time_out))

    elif type == 'mujoco':
        x_pos = info['xpos']
        cost = cost
        is_break_constraint = info['lag_cost']
        file.write("{0}, {1:.3f}, {2:.3f}\n".format(x_pos,
                                                    cost,
                                                    is_break_constraint))
    else:
        raise ValueError("Unknown type {0}".format(type))


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def bak_load_expert_data(expert_path, num_rollouts):
    expert_mean_reward = []
    for i in range(num_rollouts):
        with open(os.path.join(expert_path, "files/EXPERT/rollouts", "%s.pkl" % str(i)), "rb") as f:
            data = pickle.load(f)

        if i == 0:
            expert_obs = data['observations']
            expert_acs = data['actions']
        else:
            expert_obs = np.concatenate([expert_obs, data['observations']], axis=0)
            expert_acs = np.concatenate([expert_acs, data['actions']], axis=0)

        expert_mean_reward.append(data['rewards'])

    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = expert_obs.shape[0] / num_rollouts

    return (expert_obs, expert_acs), expert_mean_reward


def load_expert_data_tmp(expert_acs):
    expert_data = torch.load(expert_acs)
    expert_obs = []
    expert_acs = []
    for S, A in expert_data:
        for s in S:
            expert_obs += [s]
        for a in A:
            expert_acs += [a]
    expert_obs = np.array(expert_obs)
    expert_acs = np.array(expert_acs)
    # if len(expert_acs.shape) == 1:
    #     expert_acs = np.expand_dims(expert_acs, 1)
    return expert_obs, expert_acs


def load_expert_data(expert_path,
                     num_rollouts=None,
                     use_pickle5=False,
                     store_by_game=False,
                     add_next_step=False,
                     add_latent_code=False,
                     log_file=None):
    print('Loading expert data from {0}.'.format(expert_path), file=log_file, flush=True)
    file_names = sorted(os.listdir(expert_path))
    # file_names = [i for i in range(29)]
    # sample_names = random.sample(file_names, num_rollouts)
    expert_sum_rewards = []
    expert_obs = []
    expert_acs = []
    expert_rs = []
    expert_cs = []
    num_samples = 0
    if num_rollouts is None or num_rollouts > len(file_names):
        num_rollouts = len(file_names)
    for i in range(num_rollouts):
        # file_name = sample_names[i]
        file_name = file_names[i]
        with open(os.path.join(expert_path, file_name), "rb") as f:
            if use_pickle5:  # the data is stored by pickle5
                data = pickle5.load(f)
            else:
                data = pickle.load(f)
        if use_pickle5:  # for the mujoco data, observations are the original_observations
            data_obs = data['observations']
        else:
            data_obs = data['original_observations']
        data_acs = data['actions']
        if 'reward' in data.keys():
            data_rs = data['reward']
        else:
            data_rs = None
        if 'codes' in data.keys():
            data_cs = data['codes']
        else:
            data_cs = None
        if add_next_step:
            total_time_step = data_acs.shape[0] - 1
        else:
            total_time_step = data_acs.shape[0]

        if store_by_game:
            expert_obs_game = []
            expert_acs_game = []
            expert_rs_game = []
            expert_cs_game = []

        for t in range(total_time_step):
            data_obs_t = data_obs[t]
            data_ac_t = data_acs[t]
            if add_next_step:
                data_obs_next_t = data_obs[t + 1]
                data_ac_next_t = data_acs[t + 1]
            num_samples += 1
            if data_rs is not None:
                data_r_t = data_rs[t]
                if add_next_step:
                    data_r_next_t = data_rs[t + 1]
            else:
                data_r_t = 0
                if add_next_step:
                    data_r_next_t = 0
            if data_cs is not None:
                data_c_t = data_cs[t]
                if add_next_step:
                    data_c_next_t = data_c_t[t + 1]
            else:
                data_c_t = [-1]
                if add_next_step:
                    data_c_next_t = [-1]
            if add_next_step:
                data_obs_t_store = [data_obs_t, data_obs_next_t]
                data_acs_t_store = [data_ac_t, data_ac_next_t]
                data_r_t_store = [data_r_t, data_r_next_t]
                data_c_t_store = [data_c_t, data_c_next_t]
            else:
                data_obs_t_store = data_obs_t
                data_acs_t_store = data_ac_t
                data_r_t_store = data_r_t
                data_c_t_store = data_c_t
            if store_by_game:
                expert_obs_game.append(data_obs_t_store)
                expert_acs_game.append(data_acs_t_store)
                expert_rs_game.append(data_r_t_store)
                expert_cs_game.append(data_c_t_store)
            else:
                expert_obs.append(data_obs_t_store)
                expert_acs.append(data_acs_t_store)
                expert_rs.append(data_r_t_store)
                expert_cs.append(data_c_t_store)

        if store_by_game:
            expert_obs.append(np.asarray(expert_obs_game))
            expert_acs.append(np.asarray(expert_acs_game))
            expert_rs.append(np.asarray(expert_rs_game))
            expert_cs.append(np.asarray(expert_cs_game))
        if use_pickle5:  # for the mujoco data, rewards are the reward_sums
            expert_sum_rewards.append(data['rewards'])
        else:
            expert_sum_rewards.append(data['reward_sum'])
    expert_avg_sum_reward = np.mean(expert_sum_rewards)
    expert_mean_length = num_samples / len(file_names)
    print('Expert_mean_reward: {0} and Expert_mean_length: {1}.'.format(expert_avg_sum_reward, expert_mean_length),
          file=log_file, flush=True)
    if store_by_game:
        if add_latent_code:
            return (expert_obs, expert_acs, expert_rs, expert_cs), expert_avg_sum_reward
        else:
            return (expert_obs, expert_acs, expert_rs), expert_avg_sum_reward
    else:
        expert_obs = np.asarray(expert_obs)
        expert_acs = np.asarray(expert_acs)
        expert_rs = np.asarray(expert_rs)
        expert_cs = np.asarray(expert_cs)
        if add_latent_code:
            return (expert_obs, expert_acs, expert_rs, expert_cs), expert_avg_sum_reward
        else:
            return (expert_obs, expert_acs, expert_rs), expert_sum_rewards


def load_ppo_model(model_path: str, iter_msg: str, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)
    model = PPO.load(model_path)
    return model


def get_input_features_dim(feature_select_names, all_feature_names):
    if feature_select_names is None:
        feature_select_dim = None
    else:
        feature_select_dim = []
        for feature_name in feature_select_names:
            if feature_name == -1:
                feature_select_dim.append(-1)  # -1 indicates don't select
                break
            else:
                feature_select_dim.append(all_feature_names.index(feature_name))
    return feature_select_dim


def mean_std_plot_results(all_results):
    mean_results = {}
    std_results = {}
    for key in all_results[0]:
        all_plot_values = []
        max_len = 0
        min_len = float('inf')
        for results in all_results:
            plot_values = results[key]
            if len(plot_values) > max_len:
                max_len = len(plot_values)
            if len(plot_values) < min_len:
                min_len = len(plot_values)
            all_plot_values.append(plot_values)

        plot_value_all = []
        for plot_values in all_plot_values:
            plot_value_all.append(plot_values[:min_len])
        for i in range(min_len, max_len):
            plot_value_t = []
            for plot_values in all_plot_values:
                if len(plot_values) > i:
                    plot_value_t.append(plot_values[i])

            if 0 < len(plot_value_t) < len(all_plot_values):
                for j in range(len(all_plot_values) - len(plot_value_t)):
                    plot_value_t.append(plot_value_t[j % len(plot_value_t)])  # filling in values
            for j in range(len(plot_value_t)):
                plot_value_all[j].append(plot_value_t[j])
        mean_plot_values = np.mean(np.asarray(plot_value_all), axis=0)
        std_plot_values = np.std(np.asarray(plot_value_all), axis=0)
        mean_results.update({key: mean_plot_values})
        std_results.update({key: std_plot_values})

    return mean_results, std_results


def print_resource(mem_prev, time_prev, process_name, log_file, print_msg=True):
    mem_current = process_memory()
    time_current = time.time()
    print_str = "{0} consumed memory: {1:.2f}/{2:.2f} and time {3:.2f}".format(process_name,
                                                                               float(mem_current - mem_prev) / 1000000,
                                                                               float(mem_current) / 1000000,
                                                                               time_current - time_prev)
    if print_msg:
        print(print_str, file=log_file, flush=True)
        return mem_current, time_current
    else:
        return mem_current, time_current, print_str


def build_rnn_input(max_seq_length, input_data_list):
    input_seqs = []
    input_t_seq = []
    if torch.is_tensor(input_data_list[0]):
        device = input_data_list[0].device
        padding = torch.zeros([len(input_data_list[0])]).to(device)
    else:
        padding = np.zeros([len(input_data_list[0])])
    for data in input_data_list:
        if len(input_t_seq) < max_seq_length:
            input_t_seq.append(data)
            store_seq = copy.copy([padding]*(max_seq_length-len(input_t_seq)) + input_t_seq)
        else:
            input_t_seq.pop(0)
            input_t_seq.append(data)
            store_seq = copy.copy(input_t_seq)
        if torch.is_tensor(input_data_list[0]):
            input_seqs.append(torch.stack(store_seq, dim=0))
        else:
            input_seqs.append(np.stack(store_seq, dim=0))
    tmp = torch.stack(input_seqs, dim=0)
    return torch.stack(input_seqs, dim=0)


def idx2vector(indices, height, width):
    vector_all = []
    for idx in indices:
        map = np.zeros(shape=[height, width])
        if type(idx) is np.ndarray:
            # print(idx)
            x, y = np.round(idx[0], 0).astype(np.int64), np.round(idx[1], 0).astype(np.int64)
            if(type(x)!=np.int64):
                x = int(x[0]); y = int(y[0])
            else:
                x = int(x)
                y = int(y)
        else:
            x, y = int(round(idx[0], 0)), int(round(idx[1], 0))
        # if x - idx[0] != 0:
        #     print('debug')
        map[x, y] = 1  # + idx[0] - x + idx[1] - y
        vector_all.append(map.flatten())
    return np.asarray(vector_all)