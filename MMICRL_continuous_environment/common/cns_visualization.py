import itertools
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from cirl_stable_baselines3.common import callbacks
from utils.data_utils import del_and_make
from utils.model_utils import build_code, update_code
from utils.plot_utils import plot_curve


def traj_visualization_2d(config, codes, observations, save_path, model_name='', title='', axis_size=24):
    traj_num = len(observations)
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    plt.figure(figsize=(5, 5))
    for i in range(traj_num)[0: 5]:
        x = observations[i][:, config['env']["record_info_input_dims"][0]]
        y = observations[i][:, config['env']["record_info_input_dims"][1]]
        plt.plot(x, y, label='{0}th Traj, code {1}'.format(i, np.argmax(codes[i][0])))
        plt.scatter(x, y)
    xticks = np.arange(config['env']["visualize_info_ranges"][0][0],
                       config['env']["visualize_info_ranges"][0][1] + 1, 1)
    plt.xticks(xticks)
    yticks = np.arange(config['env']["visualize_info_ranges"][1][0],
                       config['env']["visualize_info_ranges"][1][1] + 1, 1)
    plt.yticks(yticks)
    # plt.yticks(config['env']["visualize_info_ranges"][1])
    # plt.xlabel(config['env']["record_info_names"][0], fontsize=axis_size)
    # plt.ylabel(config['env']["record_info_names"][1], fontsize=axis_size)
    plt.legend(fontsize=15, loc='lower right')
    plt.grid(linestyle='--')
    plt.title('{0}'.format(title), fontsize=axis_size)
    plt.savefig(os.path.join(save_path, "2d_traj_visual_{0}_{1}.png".format(model_name, title)))


def traj_visualization_1d(config, codes, observations, save_path):
    for record_info_idx in range(len(config['env']["record_info_names"])):
        plt.figure()
        plt.rcParams.update({'font.size': 16})
        plt.title("obstacle_distance_expert_demonstration")
        plt.xlabel('same_lane_leading_obstacle_distance')
        plt.ylabel('frequency')
        record_info_name = config['env']["record_info_names"][record_info_idx]
        record_obs_dim = config['env']["record_info_input_dims"][record_info_idx]
        if config['running']['store_by_game']:
            for c_id in range(config['CN']['latent_dim']):
                data_indices = np.where(np.concatenate(codes, axis=0)[:, c_id] == 1)[0]
                # tmp = np.concatenate(expert_obs, axis=0)[:, record_info_idx][data_indices]
                plt.hist(np.concatenate(observations, axis=0)[:, record_obs_dim][data_indices],
                         bins=40,
                         range=(0,95),
                         label="agent "+str(c_id))
        else:
            for c_id in range(config['CN']['latent_dim']):
                data_indices = np.where(codes[:, c_id] == 1)[0]
                plt.hist(observations[:, record_info_idx][data_indices],
                         bins=40,
                         # range=(config['env']["visualize_info_ranges"][record_info_idx][0],
                         #        config['env']["visualize_info_ranges"][record_info_idx][1])
                         )
        plt.legend()
        plt.savefig(os.path.join(save_path, "{0}_traj_visual.png".format(record_info_name)))


def constraint_visualization_2d(cost_function_with_code, feature_range, select_dims,
                                obs_dim, acs_dim, latent_dim,
                                num_points_per_feature=100,
                                axis_size=20, save_path=None, empirical_input_means=None):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size

    selected_feature_1_generation = np.linspace(feature_range[0][0], feature_range[0][1], num_points_per_feature)
    selected_feature_2_generation = np.linspace(feature_range[1][1], feature_range[1][0], num_points_per_feature)
    selected_feature_all = np.asarray(
        [d for d in itertools.product(selected_feature_1_generation, selected_feature_2_generation)])
    tmp = selected_feature_all.reshape([num_points_per_feature, num_points_per_feature, 2]).transpose(1, 0, 2)
    if empirical_input_means is None:
        input_all = np.zeros((num_points_per_feature ** 2, obs_dim + acs_dim))
    else:
        assert len(empirical_input_means) == obs_dim + acs_dim
        input_all = np.expand_dims(empirical_input_means, 0).repeat(num_points_per_feature ** 2, axis=0)
    input_all[:, select_dims[0]] = selected_feature_all[:, 0]
    input_all[:, select_dims[1]] = selected_feature_all[:, 1]
    code_axis = [0]
    for idx in range(latent_dim):
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        codes = build_code(code_axis=code_axis, code_dim=latent_dim, num_envs=1).repeat(
            repeats=num_points_per_feature ** 2, axis=0)
        code_axis[0] = update_code(code_axis[0], code_dim=latent_dim)

        with torch.no_grad():
            preds = cost_function_with_code(obs=obs, acs=acs, codes=codes)
        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.figure()
        im = plt.matshow(preds.reshape([num_points_per_feature, num_points_per_feature]).transpose(1, 0),
                         cmap='gray',  # 'cool',
                         interpolation="nearest",
                         extent=[feature_range[0][0], feature_range[0][1], feature_range[1][1], feature_range[1][0]],
                         )
        cbar = plt.colorbar(im)
        cbar.set_label("Constraint")
        plt.savefig(os.path.join(save_path, "constraint_code-{0}.png".format(codes[0])))


def constraint_visualization_1d(cost_function, feature_range, select_dim, obs_dim, acs_dim,
                                save_name, feature_data=None, feature_cost=None, feature_name=None,
                                empirical_input_means=None, num_points=1000, axis_size=24,
                                code_index=None, latent_dim=None):
    """
    visualize the constraints with partial dependency plot and (optionally) testing outputs.
    """
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    selected_feature_generation = np.linspace(feature_range[0], feature_range[1], num_points)
    if empirical_input_means is None:
        input_all = np.zeros((num_points, obs_dim + acs_dim))
    else:
        assert len(empirical_input_means) == obs_dim + acs_dim
        input_all = np.expand_dims(empirical_input_means, 0).repeat(num_points, axis=0)
    input_all[:, select_dim] = selected_feature_generation
    with torch.no_grad():
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        if code_index is not None and latent_dim is not None:
            codes = build_code(code_axis=[code_index] * len(input_all),
                               code_dim=latent_dim,
                               num_envs=len(input_all))
            preds = cost_function(obs=obs,
                                  acs=acs,
                                  codes=codes)
        else:
            preds = cost_function(obs=obs, acs=acs,
                                  force_mode='mean')  # use the mean of a distribution for visualization
    ax[0].plot(selected_feature_generation, preds, c='r', linewidth=5)
    if feature_data is not None:
        ax[0].scatter(feature_data, feature_cost)
        ax[1].hist(feature_data, bins=40, range=(feature_range[0], feature_range[1]))
        ax[1].set_axisbelow(True)
        # Turn on the minor TICKS, which are required for the minor GRID
        ax[1].minorticks_on()
        ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax[1].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
        ax[1].set_ylabel('Frequency', fontdict={'fontsize': axis_size})
    ax[0].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
    ax[0].set_ylabel('Cost', fontdict={'fontsize': axis_size})
    # ax[0].set_ylim([0, 1])
    ax[0].set_xlim(feature_range)
    ax[0].set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    fig.savefig(save_name)
    plt.close(fig=fig)


class PlotCallback(callbacks.BaseCallback):
    """
    This callback can be used/modified to fetch something from the buffer and make a
    plot using some custom plot function.
    """

    def __init__(
            self,
            train_env_id: str,
            plot_freq: int = 10000,
            log_path: str = None,
            plot_save_dir: str = None,
            verbose: int = 1,
            name_prefix: str = "model",
            plot_feature_names_dims: dict = {},
    ):
        super(PlotCallback, self).__init__(verbose)
        self.name_prefix = name_prefix
        self.log_path = log_path
        self.plot_save_dir = plot_save_dir
        self.plot_feature_names_dims = plot_feature_names_dims

    def _init_callback(self):
        # Make directory to save plots
        # del_and_make(os.path.join(self.plot_save_dir, "plots"))
        # self.plot_save_dir = os.path.join(self.plot_save_dir, "plots")
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        try:
            obs = self.model.rollout_buffer.orig_observations.copy()
        except:  # PPO uses rollout buffer which does not store orig_observations
            obs = self.model.rollout_buffer.observations.copy()
            # unormalize observations
            obs = self.training_env.unnormalize_obs(obs)
        obs = obs.reshape(-1, obs.shape[-1])  # flatten the batch size and num_envs dimensions
        rewards = self.model.rollout_buffer.rewards.copy()
        for record_info_name in self.plot_feature_names_dims.keys():
            plot_record_infos, plot_costs = zip(
                *sorted(zip(obs[:, self.plot_feature_names_dims[record_info_name]], rewards)))
            path = os.path.join(self.plot_save_dir, f"{self.name_prefix}_{self.num_timesteps}_steps")
            if not os.path.exists(path):
                os.mkdir(path)
            plot_curve(draw_keys=[record_info_name],
                       x_dict={record_info_name: plot_record_infos},
                       y_dict={record_info_name: plot_costs},
                       xlabel=record_info_name,
                       ylabel='cost',
                       save_name=os.path.join(path, "{0}_visual.png".format(record_info_name)),
                       apply_scatter=True
                       )
        # self.plot_fn(obs, os.path.join(self.plot_save_dir, str(self.num_timesteps) + ".png"))